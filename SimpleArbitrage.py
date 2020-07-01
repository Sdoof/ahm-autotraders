from enum import Enum
from fmclient import Agent, OrderSide, Order, OrderType
import fmclient.fmio.net.fmapi.rest.request as request
from fmclient.utils import constants as cons
import copy, math, time, traceback

PROFIT_MARGIN = 10

# bot role enumeration
class Role(Enum):
    BUYER = 0,
    SELLER = 1

# bot type enumeration
class BotType(Enum):
    MARKET_MAKER = 0,
    REACTIVE = 1


class DSBot(Agent):
    """
    Simple stupid arbitrage bot but it does its job and does it well
    """
    def __init__(self, account, email, password, marketplace_id, bot_type):
        super().__init__(account, email, password, marketplace_id, name="DSBot")
        self._market_id = -1
        self._public_market_id = 0
        self._private_market_id = 0
        self._role = None
        # bot_type state storage
        self._bottype = bot_type

        # PERFORMANCE OPTIMISATIONS
        cons.ASYNCIO_MAX_THREADS = 4
        request.concurrency = 24
        cons.MONITOR_ORDER_BOOK_DELAY = 0.25
        cons.MONITOR_HOLDINGS_DELAY = 0.25
        cons.WS_SEND_DELAY = 36000
        cons.WS_LISTEN_DELAY = 36000
        cons.WS_MESSAGE_DELAY = 36000

        # CONSTANTS
        # session length (in seconds)
        self._SESSION_LENGTH = 600
        # max market price
        self._MAX_MKT_PRICE = 1000
        # assets required based on extrinsic criteria
        self._ASSETS_REQ = 20
        # dealer ID in private market
        self._DEALER_ID = 'M000'

        # TRADING LOGIC
        # a higher aggresion level (>0, default 0) aims to increase profit per trade, but may initially decrease trade success rate
        self._INITIAL_AGGRESSION = 25
        # number of state refreshes which one market maker position persists for
        self._MM_REFRESH_INTERVAL = 32
        # number of state refreshes which one reactive position persists for
        self._REACTIVE_REFRESH_INTERVAL = 4

        # STATE STORAGE
        # start time
        self._start_time = time.time()
        # public order management
        self._active_orders = {}
        self._active_order_age = {}
        self._pending_orders = {}
        # private market info
        self._private_price = self._MAX_MKT_PRICE/2
        self._private_units = 0
        # asset state
        self._current_assets = self._ASSETS_REQ
        # target profit, scaled using aggression
        self._target_profit = int(round(PROFIT_MARGIN + PROFIT_MARGIN * self._INITIAL_AGGRESSION))


    def role(self):
        return self._role

    def bot_type(self):
        return self._bottype
    
    def time_elapsed(self):
        # time elapsed since the bot was initialised
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
        self._active_order_age[order.market_id] = 0
        # if the order isn't a cancel order, set it as the active order
        if order.type != OrderType.CANCEL:
            self._active_orders[order.market_id] = order
            self.inform(f"[ORDER ACCEPTED]{(' PRIVATE ' if order.market_id == self._private_market_id else ' ')}{order.side.name} order @ {order.price}.")
        # if accepted order is for cancellation, clear active order
        else:
            self._active_orders[order.market_id] = None
        self._pending_orders[order.market_id] = None

    def order_rejected(self, info, order):
        # clear pending order and inform order rejection
        self._pending_orders[order.market_id] = None
        self.inform(f"[ORDER REJECTED] {info}, for{(' PRIVATE ' if order.market_id == self._private_market_id else ' ')}{order.side.name} order @ {order.price}.")


    def received_order_book(self, order_book, market_id):
        try:
            # order housekeeping
            try:
                # increment order age
                updated_ao = [o for o in order_book if o.mine][0]
                if self._active_orders.get(market_id) == updated_ao:
                    self._active_order_age[market_id] = self._active_order_age.get(market_id, 0) + 1
                # unless the order is new, then make its age 1
                else:
                    self._active_orders[market_id] = updated_ao
                    self._active_order_age[market_id] = 1
                # purge order if it has stayed in the market for too long
                refresh_interval = (self._MM_REFRESH_INTERVAL if self.bot_type() == BotType.MARKET_MAKER else self._REACTIVE_REFRESH_INTERVAL)
                if self._active_order_age.get(market_id, 0) > refresh_interval:
                    self.cancel_order(self._active_orders[market_id])
                    self.inform("[ORDER REFRESH] purging stagnant order.")
            # clear active order info if no active order is found
            except IndexError:
                self._active_orders[market_id] = None
                self._active_order_age[market_id] = 0
            
            # public market actions
            if market_id == self._public_market_id:
                # stop processing if orderbook doesn't contain orders or role is not defined
                if len([o for o in order_book if not o.mine]) == 0 or self.role() == None:
                    return
                
                # if orderbook contains orders, initialise relevant variables (profit/critical prices)
                self.update_aggression()
                # the price at which profit will be made according to the minimum profit margin
                buy_profit = self._private_price - PROFIT_MARGIN
                sell_profit = self._private_price + PROFIT_MARGIN
                # the critical price at which desired profit (scaled using aggression) will be made
                buy_crit = self._private_price - self._target_profit
                sell_crit = self._private_price + self._target_profit
                # inform bot's desired market price
                if self.role() != None:
                    self.inform(f"[{self.role().name}] target price: {buy_crit if self.role() == Role.BUYER else sell_crit}.")
                # intialise order
                new_order = Order(self._private_price, 1, OrderType.LIMIT, (OrderSide.BUY if self.role() == Role.BUYER else OrderSide.SELL), self._public_market_id, ref='pub_order')
                
                # implementation for market maker and reactive bot types
                # set appropriate buyer price, capture arbitrage if reactive
                if self.role() == Role.BUYER:
                    # find ask and assign order price so that profit will be made
                    ask = min([o.price for o in order_book if o.side == OrderSide.SELL and not o.mine] + [self._MAX_MKT_PRICE])
                    new_order.price = buy_crit
                    # identify all profitable trade opportunities
                    if ask <= buy_profit:
                        self._print_trade_opportunity(f'BUY @ {ask}.')
                        # reactive bot: capture desirable arbitrage opportunity at market price
                        if self.bot_type() == BotType.REACTIVE and ask <= buy_crit:
                            new_order.price = ask
                            self.send_if_valid_order(new_order)
                        if ask >= buy_crit:
                            self.inform(f"[{self.bot_type().name}] order is profitable but not profitable enough.")
                
                # set appropriate seller price, capture arbitrage if reactive
                elif self.role() == Role.SELLER:
                    # find bid and assign order price so that profit will be made
                    bid = max([o.price for o in order_book if o.side == OrderSide.BUY and not o.mine] + [0])
                    new_order.price = sell_crit
                    # identify all profitable trade opportunities
                    if bid >= sell_profit:
                        self._print_trade_opportunity(f'SELL @ {bid}.')
                        # reactive bot: capture desirable arbitrage opportunity at market price
                        if self.bot_type() == BotType.REACTIVE and bid >= sell_crit:
                            new_order.price = bid
                            self.send_if_valid_order(new_order)
                        if bid <= sell_crit:
                            self.inform(f"[{self.bot_type().name}] order is profitable but not profitable enough.")
                
                # market maker bot: send order if the order is valid, regardless if arbitrage opportunity is present
                if self.bot_type() == BotType.MARKET_MAKER:
                    self.send_if_valid_order(new_order)

            # private market actions
            elif market_id == self._private_market_id:
                try:
                    # infer bot role from private market order
                    po = [o for o in order_book if not o.mine][0]
                    if po.side == OrderSide.BUY:
                        self._role = Role.BUYER
                    elif po.side == OrderSide.SELL:
                        self._role = Role.SELLER
                    # store private market price and units info
                    self._private_price = int(po.price)
                    self._private_units = po.units
                except IndexError:
                    # on role detection failure, clear role, private price and private units
                    self._role = None
                    self._private_price = self._MAX_MKT_PRICE/2
                    self._private_units = 0
        # display tracebacks for main recurring loops to aid debugging
        except:
            traceback.print_exc()


    def _print_trade_opportunity(self, other_order):
        self.inform("I am a {0} with profitable order {1}".format(self.role().name, other_order))

    def received_completed_orders(self, orders, market_id=None):
        pass


    def received_holdings(self, holdings):
        try:
            self._current_assets = sum([m['units'] for m in holdings['markets'].values()])
            # if total assets do not match required assets
            if self._current_assets != self._ASSETS_REQ and self._private_units > 0:
                # trade from private market until required asset count is reached again
                private_order = Order(self._private_price, 1, OrderType.LIMIT, (OrderSide.BUY if self._current_assets < self._ASSETS_REQ else OrderSide.SELL), self._private_market_id, ref='priv_order')
                private_order.owner_or_target = self._DEALER_ID
                # decrement private units and disable role if private units have reached zero (no arbitrage opportunities left)
                self._private_units -= 1
                if self._private_units == 0:
                    self._role = None
                self.send_if_valid_order(private_order)
        # display tracebacks for main recurring loops to aid debugging
        except:
            traceback.print_exc()


    def received_marketplace_info(self, marketplace_info):
        pass
    

    def send_if_valid_order(self, order):
        # ensure only one order can be active
        if self._active_orders.get(order.market_id) != None or self._pending_orders.get(order.market_id) != None:
            return False
        # ensure that price is valid
        order.price = max(min(order.price, self._MAX_MKT_PRICE), 0)
        # public orders can only be sent when arbitrage is not currently in progress
        # private orders can only be sent to correct uneven asset counts
        if (self._current_assets == self._ASSETS_REQ) == (order.market_id == self._private_market_id):
            return False
        # check if sufficient capital
        capital = False
        if order.side == OrderSide.BUY:
            capital = self._holdings['cash']['available_cash'] >= order.price * order.units
        elif order.side == OrderSide.SELL:
            capital = self._holdings['markets'][order.market_id]['available_units'] >= order.units
        if not capital:
            self.inform("[ORDER NOT SENT] insufficient capital.")
            return False
        # send order if valid
        self.inform(f"[{self.bot_type().name}] sending valid{(' PRIVATE ' if order.market_id == self._private_market_id else ' ')}{order.side.name} order @ {order.price}.")
        self._pending_orders[order.market_id] = order
        self.send_order(order)
        return True


    def cancel_order(self, order):
        cancel_order = copy.copy(order)
        cancel_order.type = OrderType.CANCEL
        cancel_order.ref = f'{order.ref}_cancel'
        self._pending_orders[order.market_id] = cancel_order
        self.send_order(cancel_order)

    def update_aggression(self):
        x = self.time_elapsed() / self._SESSION_LENGTH
        # aggression decays at a initially steeper and eventually gentler rate as the session progresses
        y = -math.sin(x * math.pi / 2) + 1
        # as the session progresses, the bot gradually becomes more conservative, being more accepting of lower returns
        self._target_profit = int(round(PROFIT_MARGIN + PROFIT_MARGIN * y * self._INITIAL_AGGRESSION))

    def run(self):
        self.initialise()
        self.start()


if __name__ == '__main__':
    # trading account details
    FM_ACCOUNT = ''
    FM_EMAIL = ''
    FM_PASSWORD = ''
    MARKETPLACE_ID = 0
    BOT_TYPE = BotType.REACTIVE

    ds_bot = DSBot(FM_ACCOUNT, FM_EMAIL, FM_PASSWORD, MARKETPLACE_ID, BOT_TYPE)
    ds_bot.run()