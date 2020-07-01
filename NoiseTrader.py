from enum import Enum
from fmclient import Agent, OrderSide, Order, OrderType
from fmclient.utils import constants as cons
import fmclient.fmio.net.fmapi.rest.request as request
import copy, random, time, traceback
from scipy.stats import t

PROFIT_MARGIN = 10

# enum for the roles of the bot
class Role(Enum):
    BUYER = 0,
    SELLER = 1

# enum for the type of bot
class BotType(Enum):
    MARKET_MAKER = 0,
    REACTIVE = 1


class DSBot(Agent):
    def __init__(self, account, email, password, marketplace_id, bot_type):
        super().__init__(account, email, password, marketplace_id, name="DSBot")
        self._market_id = -1
        self._public_market_id = 0
        self._private_market_id = 0
        self._role = None
        self._bottype = bot_type
        self._start_time = time.time()
        
        # CONSTANTS
        # max market price
        self._MAX_MKT_PRICE = 1000
        # assets required based on extrinsic criteria
        self._ASSETS_REQ = 10
        # maximum active orders
        self._MAX_ACTIVE_ORDERS = 1
        # TRADE LOGIC
        # time decay factor, value * time_decay_factor ^ time_elapsed_since_ob
        self._TIME_DECAY_FACTOR = 0.8
        # confidence level of interval; lower confidence leads to more completed trades with smaller profit per trade
        self._CONFIDENCE = 0.6
        # number of state refreshes which one position persists for
        self._ORDER_REFRESH_INTERVAL = 8
        # PERFORMANCE
        # max observations in orderbook history dataset
        self._MAX_OBS = 500

        # STATE STORAGE
        # orders
        self._order_age = {}
        self._own_orders = {}
        self._pending_orders = {}
        # known holdings
        self._known_holdings = {}
        # orderbook history
        self._ob_data = []
        self._obs = 0

        # TRADING LOGIC
        self._mu_sum = 0
        self._eprice = self._MAX_MKT_PRICE/2
        self._evol = self._MAX_MKT_PRICE - 1

        # PERFORMANCE OPTIMISATIONS
        cons.ASYNCIO_MAX_THREADS = 4
        request.concurrency = 24
        cons.MONITOR_ORDER_BOOK_DELAY = 0.2
        cons.MONITOR_HOLDINGS_DELAY = 0.2
        cons.MONITOR_SESSION_DELAY = 0.2
        cons.WS_SEND_DELAY = 36000
        cons.WS_LISTEN_DELAY = 36000
        cons.WS_MESSAGE_DELAY = 36000

    def role(self):
        return self._role

    def bot_type(self):
        return self._bottype

    def time_elapsed(self):
        return time.time() - self._start_time

    def initialised(self):
        # assign private and public market IDs
        for m_id, m_info in self.markets.items():
            if m_info['privateMarket']:
                self._private_market_id = m_id
                self.inform(f"[PRIVATE MARKET] {m_info['name']}, ID: {m_id}")
            else:
                self._public_market_id = m_id
                self.inform(f"[PUBLIC MARKET] {m_info['name']}, ID: {m_id}")
        self.inform('Bot initialised.')

    def order_accepted(self, order):
        self._pending_orders.pop(order.ref)

    def order_rejected(self, info, order):
        self.inform(f"[REJECTED] {info}, for order: {order.ref}.")
        self._pending_orders.pop(order.ref)

    def order_housekeeping(self, order_book):
        # count own orders
        self._own_orders = {o.ref:o for o in order_book if o.mine}
        
        # monitor age of orders and cancel stagnant orders
        if len(self._own_orders) > 0:
            # increment order age, and purge nonexistent orders
            new_order_age = {}
            # THIS CODE DOESN'T WORK FOR MULTIPLE SAME-PRICE ORDERS: FLEXEMARKETS STACKS MULTIPLE ORDERS OF THE SAME PRICE AND GIVES IT A NEW REF
            for r, o in self._own_orders.items():
                new_order_age[r] = self._order_age.get(r, 0) + 1
            self._order_age = new_order_age

            # check for and cancel stagnant orders
            for r, o in self._order_age.items():
                if o > self._ORDER_REFRESH_INTERVAL and len(self._pending_orders) == 0:
                    cancel_order = copy.copy(self._own_orders[r])
                    cancel_order.type = OrderType.CANCEL
                    cancel_order.ref = f'{r}_cancel'
                    self._pending_orders[cancel_order.ref] = cancel_order
                    self.send_order(cancel_order)
                    self.inform("[ORDER REFRESH] purging stagnant order.")

    def record_ob_data(self, order_book):
        # record market history data
        try:
            buys = [o.price for o in order_book if o.side == OrderSide.BUY and not o.mine] + [0]
            sells = [o.price for o in order_book if o.side == OrderSide.SELL and not o.mine] + [self._MAX_MKT_PRICE]
            spread = min(sells) - max(buys)
            mu = (max(buys) + min(sells)) / 2
            self._ob_data.append([time.time(), mu, spread])
            self._mu_sum += mu
        except Exception as e:
            self.inform(f"[PARSE FAILED] {e}, for orderbook: {order_book}.")

        # limit max observations for performance reasons
        if self._obs >= self._MAX_OBS:
            self._mu_sum -= self._ob_data[0][1]
            self._ob_data = self._ob_data[1:]
        else:
            self._obs += 1

    def compute_confint(self):
        # compute confint from ob_data
        total = 0.0
        sdtotal = 0.0
        n = 0.0
        # compute eprice and evol, weighted based on time
        for v in self._ob_data:
            weight = self._TIME_DECAY_FACTOR ** (time.time() - v[0])
            n += weight
            total += v[1] * weight
            sdtotal += ((v[1] - self._mu_sum / self._obs) ** 2) * weight
        self._eprice = total / n
        self._evol = sdtotal / n

        # if evol is zero, no activity is occuring; discourage bot from trading due to lack of liquidity
        if self._evol < 1:
            self._evol = (self._MAX_MKT_PRICE - 1) ** 2
        
        # CI based on Student t distribution
        return t.interval(self._CONFIDENCE, int(round(n)), self._eprice, self._evol**0.5)

    def check_order_validity(self, order, profit):
        # order validity (profitable and sufficient capital)
        # profit check
        valid_order = profit > PROFIT_MARGIN
        # capital check
        if order.side == OrderSide.BUY:
            valid_order = valid_order and self._holdings['cash']['available_cash'] > order.price * order.units
        elif order.side == OrderSide.SELL:
            valid_order = valid_order and self._holdings['markets'][order.market_id]['available_units'] > order.units
        return valid_order

    def received_order_book(self, order_book, market_id):
        try:
            # PUBLIC MARKET ACTIONS
            if self._public_market_id == market_id and len(order_book)>0:
                self.order_housekeeping(order_book)

                # record and compute confint if market info is available
                ci = (0, self._MAX_MKT_PRICE)
                if len([o for o in order_book if not o.mine])>0:
                    self.record_ob_data(order_book)
                    ci = self.compute_confint()
                
                # MARKET MAKER STRATEGY
                if self.bot_type() == BotType.MARKET_MAKER:
                    # if no order exists in the market, create new order
                    if len(self._own_orders) < self._MAX_ACTIVE_ORDERS and len(self._pending_orders) == 0:
                        # initialise order
                        new_order = Order(0, 1, OrderType.LIMIT, OrderSide.BUY, self._public_market_id, ref='pub_order')
                        
                        # sell if too many assets, buy if too few assets, choose based on market price if asset number is correct
                        is_sell = self._ob_data[-1][1] / self._MAX_MKT_PRICE
                        if self._holdings['markets'][market_id]['units'] > self._ASSETS_REQ:
                            is_sell = 1
                        elif self._holdings['markets'][market_id]['units'] < self._ASSETS_REQ:
                            is_sell = 0
                        
                        # create order based on CI
                        self.inform(f"[PUBLIC] w{int(round(self._CONFIDENCE*100))}CI: {tuple([round(v, 3) for v in ci])}")
                        if is_sell < 0.5:
                            new_order.price = int(round(max(0, ci[0])))
                        else:
                            new_order.side = OrderSide.SELL
                            new_order.price = int(round(min(ci[1], self._MAX_MKT_PRICE)))

                        # check that order is valid before sending
                        if self.check_order_validity(new_order, (ci[1] - ci[0])/2):
                            self.inform(f"[{self.bot_type().name}] sending valid {new_order.side.name} order @ {new_order.price}.")
                            self._pending_orders[new_order.ref] = new_order
                            self.send_order(new_order)
                
            # infer bot role from private market order
            ## THIS ASSUMES THERE IS ONLY 1 ORDER IN THE PRIVATE MARKET AND NO NEW ORDERS ARRIVE THROUGH THE SESSION
            if self.role() == None and market_id == self._private_market_id and len(order_book)>0:
                if order_book[0].side == OrderSide.BUY:
                    self._role = Role.BUYER
                else:
                    self._role = Role.SELLER
        except:
            traceback.print_exc()

    def _print_trade_opportunity(self, other_order):
        self.inform("I am a {} with profitable order {}".format(self.role().name, other_order))

    def received_completed_orders(self, orders, market_id=None):
        self.inform("[COMPLETED ORDERS] market {}: {}".format(market_id, orders))

    def received_holdings(self, holdings):
        if holdings != self._known_holdings:
            # update holdings and trigger changed event
            self.holdings_changed(self._known_holdings, holdings)
            self._known_holdings = holdings

    def holdings_changed(self, old_holdings, new_holdings):
        # inform a change in holdings
        holdings_str = '${}/${}'.format(new_holdings['cash']['available_cash'], new_holdings['cash']['cash'])
        for m_id, m_info in new_holdings['markets'].items():
            holdings_str = holdings_str +  ' | {}: {}/{}'.format(m_id, m_info['available_units'], m_info['units'])
        # self.inform(f"[HOLDINGS CHANGE] {holdings_str}")

    def received_marketplace_info(self, marketplace_info):
        self.inform(f"[MARKET INFO] {marketplace_info}")

    def run(self):
        self.initialise()
        self.start()


if __name__ == '__main__':
    FM_ACCOUNT = ''
    FM_EMAIL = ''
    FM_PASSWORD = ''
    MARKETPLACE_ID = 0
    BOT_TYPE = BotType.MARKET_MAKER

    ds_bot = DSBot(FM_ACCOUNT, FM_EMAIL, FM_PASSWORD, MARKETPLACE_ID, BOT_TYPE)
    ds_bot.run()
