from fmclient import Agent
from fmclient import Order, OrderSide, OrderType
import fmclient.fmio.net.fmapi.rest.request as request
from fmclient.utils import constants as cons

import collections, copy, time
import numpy as np
from scipy.stats import t

class CAPMBot(Agent):
    """
    Does statistical arbitrage.
    """
    def __init__(self, account, email, password, marketplace_id, risk_penalty=0.01, session_time=20):
        super().__init__(account, email, password, marketplace_id, name="CAPM Bot")
        self._payoffs = {}
        self._risk_penalty = risk_penalty
        self._session_time = session_time
        self._market_ids = {}
        self._e_payoffs = {}
        self._cov = []
        self._start_time = time.time()

        # CONSTANTS
        # number of state refreshes which one position persists for
        self._REFRESH_INTERVAL = 16
        # decay factor used for time-weighted variance computations
        self._TIME_WEIGHTED_DECAY = 0.9
        # confidence level required before offloading assets for arbitrage liquidity
        self._ARBITRAGE_CONDFIDENCE = 0.95
        # speed at which orders are sent (per second)
        self._ORDER_SEND_SPEED = 4

        # ORDER STATE
        # nested dicts store up to one buy and one sell position, for each market
        nested_dict = lambda: collections.defaultdict(nested_dict)
        self._active_orders = nested_dict()
        self._active_order_age = nested_dict()
        self._pending_orders = nested_dict()

        # ORDER QUEUES
        # orders are buffered in the queue after verification and interval-fed to server
        self._cancel_queue = []
        self._order_queue = []

        # STATE STORAGE
        # total count-only holdings
        self._total_holdings = {}
        # most recent bid and ask prices
        self._bids = {}
        self._asks = {}
        # time-weighted variance
        self._timewvar = nested_dict()
        # order objects with bid/ask prices that will increase market performance
        self._crit_orders = nested_dict()

    def time_elapsed(self):
        # time elapsed since the bot was initialised
        return time.time() - self._start_time
    
    def initialised(self):
        payoff_matrix = []
        for market_id, market_info in self.markets.items():
            security = market_info["item"]
            description = market_info["description"]
            self._payoffs[security] = [int(a) for a in description.split(",")]
            payoff_matrix.append(self._payoffs[security])
            # store expected payoffs of each security
            self._e_payoffs[market_id] = np.mean(self._payoffs[security])
            # store market IDs
            self._market_ids[security] = market_id
        # generate covariance matrix
        self._cov = np.cov(payoff_matrix)

    def calc_timewvar(self, price_t, price_t1, var_t1=None, decay=None):
        """
        Returns time-weighted variance of a time series, given the variance of the previous period, current price, and previous period price.
        """
        decay = decay if decay else self._TIME_WEIGHTED_DECAY
        if var_t1!=None:
            return (1 - decay)*(price_t-price_t1)**2 + decay*var_t1
        else:
            return (price_t-price_t1)**2
    
    def t_confint_bound(self, mu, sigma, confidence, n, upper=False):
        """
        Returns the int-rounded lower or upper bound of a confidence interval based on the Student t distribution.
        """
        if sigma==0:
            return mu
        else:
            return int(round(t.interval(confidence, n, mu, sigma)[int(upper)]))

    def performance(self, holdings=None):
        """
        Returns performance score given holdings.
        """
        # get current performance if no holdings object is defined
        holdings = holdings if holdings else self._holdings
        # compute dollar value for asset holdings
        asset_dollars = [holdings['cash']['cash']] + [self._e_payoffs[m_id] * m['units'] for m_id, m in holdings['markets'].items()]
        # e(payoff) = cash + sum(e(payoff) for all assets)
        e_payoff = sum(asset_dollars)
        # PF variance = weighted sum of covariances
        w = [v/e_payoff for v in asset_dollars[1:]]
        pf_variance = np.matmul(w, np.matmul(self._cov, w))
        # performance = E(payoff) - risk_penalty * variance
        return e_payoff - self._risk_penalty * pf_variance
    
    def get_potential_performance(self, orders):
        """
        Returns potential performance from current state, given a list of orders to execute.
        """
        # get potential performance
        potential_holdings = copy.deepcopy(self._holdings)
        # perform potential changes on holdings based on order list
        for o in orders:
            if o.type==OrderType.LIMIT:
                buy = 1 if o.side==OrderSide.BUY else -1
                potential_holdings['cash']['cash'] -= buy*o.price*o.units
                potential_holdings['markets'][o.market_id]['units'] += buy*o.units
        return self.performance(potential_holdings)

    def get_performance_change(self, orders):
        return self.get_potential_performance(orders) - self.performance()

    def is_portfolio_optimal(self):
        """
        Returns a boolean indicating if PF is optimal, given current composition and market prices.
        """
        # is portfolio optimal?
        assert len(self._crit_orders) == len(self._bids) == len(self._asks)
        optimal = True
        for m_id, bid in self._bids.items():
            optimal &= bid < self._crit_orders.get(m_id, {}).get(OrderSide.SELL).price
        for m_id, ask in self._asks.items():
            optimal &= ask > self._crit_orders.get(m_id, {}).get(OrderSide.BUY).price
        return optimal

    def find_crit_order(self, market_id, side, holdings=None, units=1):
        """
        Returns a buy/sell order with score-increasing critical price, for the chosen market.
        """
        holdings = holdings if holdings else self._holdings
        # initialise test order and relevant variables
        test_o = Order(0, units, OrderType.LIMIT, side, market_id, ref=f"crit_{market_id}_{side.name}")
        buy = side==OrderSide.BUY
        sell = not buy
        x = 1 if sell else -1

        # if order exists, iterate on current price until critical price is reached
        current_crit_o = self._crit_orders.get(market_id, {}).get(side)
        if current_crit_o:
            test_o.price = copy.copy(current_crit_o.price)
            while self.get_performance_change([test_o]) <= 0:
                test_o.price += x
            while self.get_performance_change([test_o]) > 0:
                test_o.price -= x
            test_o.price += x
            return test_o
        
        # else, use binary search to find solution
        else:
            price_range = [self.markets[market_id]['minimum'], self.markets[market_id]['maximum']]
            # solve inequality: price @ min(pchange), s.t. pchange>0
            while price_range[1] - price_range[0] > 1:
                test_o.price = int(round(np.mean(price_range)))
                pchange = self.get_performance_change([test_o])
                if ((pchange>0) & buy) | ((pchange<0) & sell):
                    price_range = [test_o.price, price_range[1]]
                elif ((pchange>0) & sell) | ((pchange<0) & buy):
                    price_range = [price_range[0], test_o.price]
                # if performance change equals zero, score-increasing order can be immediately inferred
                else:
                    test_o.price = test_o.price + x
                    return test_o
            test_o.price = price_range[int(sell)]
            return test_o

    def refresh_crit_orders(self, market_id):
        """
        Refreshes the critical orders for a given market.
        """
        new_crit_orders = {s:self.find_crit_order(market_id, s) for s in [OrderSide.BUY, OrderSide.SELL]}
        # track previous prices and purge market orders whose prices have changed
        for s,o in new_crit_orders.items():
            existing_o = self._crit_orders.get(market_id,{}).get(s)
            if existing_o:
                if o.price!=existing_o.price:
                    active_order = self._active_orders.get(market_id, {}).get(s)
                    if active_order:
                        self.cancel_order(active_order)
                    self.inform(f"[{self.markets[o.market_id]['item']}] new critical {o.side.name} price: {o.price}.")
            self._crit_orders[market_id][s] = o

    def enforce_liquidity(self, market_id, units=1, confidence=None):
        """
        If a sell @ bid in the current market leads to net improved performance, creates and sends the sell @ bid order.
        Creates confidence bounds for bids/asks based on market volatility, to ensure arbitrage execution.
        """
        confidence = confidence if confidence else self._ARBITRAGE_CONDFIDENCE
        # if volatility info is not available, end function
        ask_vol = self._timewvar.get(market_id,{}).get(OrderSide.SELL)
        if ask_vol==None: return

        # initialise buy order and cash needed for the order
        ask = self.t_confint_bound(self._asks[market_id], ask_vol**0.5, confidence, int(self.time_elapsed()), upper=True)
        if not (self.markets[market_id]['minimum'] <= ask <= self.markets[market_id]['maximum']): return
        current_buy = Order(ask, units, OrderType.LIMIT, OrderSide.BUY, market_id, ref="current_buy")
        cn = max(ask*units - self._holdings['cash']['available_cash'], 0)
        # if more than enough cash is available, end function
        if not cn: return
        # else, if another market can sell @ bid and improve net performance buying in current market @ ask, sell @ bid in that market
        highest_pchange = 0
        highest_sell = None
        for other_m_id, raw_bid in {k:v for k,v in self._bids.items() if (k!=market_id) & (v>self.markets[k]['minimum'])}.items():
            # if volatility info is not available, continue to next market
            bid_vol = self._timewvar.get(market_id,{}).get(OrderSide.BUY)
            if bid_vol==None: continue
            # if bid price would create an invalid order, continue to next market
            bid = self.t_confint_bound(raw_bid, bid_vol**0.5, confidence, int(self.time_elapsed()))
            if not (self.markets[other_m_id]['minimum'] <= bid <= self.markets[other_m_id]['maximum']): continue
            # initialise the sell @ bid order
            other_sell = Order(bid, int(cn/bid)+(cn%bid>0), OrderType.LIMIT, OrderSide.SELL, other_m_id, ref="other_sell")
            # records the (valid) order pair with the highest performance change
            if self.valid_order(other_sell, bypass_tracking=True):
                pchange = self.get_performance_change([current_buy, other_sell])
                if pchange > highest_pchange:
                    highest_sell = other_sell
                    highest_pchange = pchange
        
        # sell @ bid in the market that results in the highest performance increase
        if highest_sell:
            if self.valid_order(highest_sell, bypass_tracking=True):
                self.send_order(highest_sell)
                self._holdings['markets'][highest_sell.market_id]['available_units'] -= highest_sell.units
                self.inform(f"[LIQUIDITY MANAGEMENT] offloading {self.markets[highest_sell.market_id]['item']} @ {highest_sell.price}.")
    
    def order_accepted(self, order):
        self._active_order_age[order.market_id][order.side] = 0
        # if the order isn't a cancel order, set it as the active order
        if order.type != OrderType.CANCEL:
            self._active_orders[order.market_id][order.side] = order
        # if accepted order is for cancellation, clear active order
        else:
            self._active_orders[order.market_id][order.side] = None
        self._pending_orders[order.market_id][order.side] = None

    def order_rejected(self, info, order):
        # clear pending order and inform order rejection
        self._pending_orders[order.market_id][order.side] = None
        self.inform(f"[ORDER REJECTED] {info}, for {self.markets[order.market_id]['item']} {order.side.name} order @ {order.price}.")

    def order_housekeeping(self, order_book, market_id):
        """
        Observes market orders, records relevant data, cleans stagnant active orders.
        """
        # market order management for proactive orders
        updated_aos = [o for o in order_book if o.mine]
        old_aos_market = copy.deepcopy(self._active_orders.get(market_id,{}))
        old_ao_age_market = copy.deepcopy(self._active_order_age.get(market_id,{}))
        self._active_orders[market_id] = {OrderSide.BUY:None, OrderSide.SELL:None}
        self._active_order_age[market_id] = {OrderSide.BUY:0, OrderSide.SELL:0}
        for o in updated_aos:
            self._active_orders[market_id][o.side] = o
            # increment order age
            if old_aos_market.get(o.side):
                self._active_order_age[market_id][o.side] = old_ao_age_market.get(o.side, 0) + 1
            # unless the order is new, then make its age 1
            else:
                self._active_order_age[market_id][o.side] = 1
            # purge order if it has stayed in the market for too long
            if self._active_order_age.get(market_id, {}).get(o.side, 0) > self._REFRESH_INTERVAL:
                self.cancel_order(o)

        # send an order from the cancel queue
        if len(self._cancel_queue) > 0:
            self.send_order(self._cancel_queue.pop(0))

        # obtain bid/ask prices
        new_bid = max([o.price for o in order_book if o.side==OrderSide.BUY and not o.mine] + [self.markets[market_id]['minimum']])
        new_ask = min([o.price for o in order_book if o.side==OrderSide.SELL and not o.mine] + [self.markets[market_id]['maximum']])
        
        # record time-weighted variance
        if self._bids.get(market_id)!=None:
            self._timewvar[market_id][OrderSide.BUY] = self.calc_timewvar(new_bid, self._bids[market_id], var_t1=self._timewvar.get(market_id,{}).get(OrderSide.BUY))
        if self._asks.get(market_id)!=None:
            self._timewvar[market_id][OrderSide.SELL] = self.calc_timewvar(new_ask, self._asks[market_id], var_t1=self._timewvar.get(market_id,{}).get(OrderSide.SELL))
        self._bids[market_id] = new_bid
        self._asks[market_id] = new_ask

    def received_order_book(self, order_book, market_id):
        # verify order book is properly formatted
        if type(order_book) != list: return
        self.order_housekeeping(order_book, market_id)

        # enforce liquidity for performance-increasing orders
        self.enforce_liquidity(market_id) 
        # create and send critical orders
        for o in self._crit_orders.get(market_id,{}).values():
            self.send_if_valid_order(o)

    def received_holdings(self, holdings):
        # verify holdings are properly formatted
        if type(holdings) != dict: return

        # if total holdings have changed, trigger changed event
        updated_total_holdings = {m:v['units'] for m,v in holdings['markets'].items()}
        updated_total_holdings['cash'] = holdings['cash']['cash']
        if updated_total_holdings != self._total_holdings:
            self.total_holdings_changed(updated_total_holdings, self._total_holdings)
            self._total_holdings = updated_total_holdings
        
        # send first valid order from the order queue
        for _ in range(self._ORDER_SEND_SPEED-1):
            if len(self._order_queue) > 0:
                unsent = True
                i = 0
                while (unsent) & (i<len(self._order_queue)):
                    if self.valid_order(self._order_queue[i], bypass_tracking=True):
                        o = self._order_queue.pop(i)
                        self.send_order(o)
                        # temporarily update holdings to maintain consistency till next refresh
                        if o.side==OrderSide.BUY:
                            self._holdings['cash']['available_cash'] -= o.price*o.units
                        elif o.side==OrderSide.SELL:
                            self._holdings['markets'][o.market_id]['available_units'] -= o.units
                        unsent = False
                    i += 1
            time.sleep(1/self._ORDER_SEND_SPEED)

    def total_holdings_changed(self, new_total_holdings, old_total_holdings):
        """
        Occurs when total units/cash held have changed.
        """
        # refresh critical order data
        for m_id in self.markets.keys():
            self.refresh_crit_orders(m_id)
        self.inform(f"[PERFORMANCE] current performance: {round(self.performance(), 1)}.")

    def received_marketplace_info(self, marketplace_info):
        pass

    def queue_order(self, order):
        if order.type==OrderType.CANCEL:
            self._cancel_queue.append(order)
        else:
            self._order_queue.append(order)

    def cancel_order(self, order):
        """
        Cancels an order using the cancel queue.
        """
        cancel_order = copy.deepcopy(order)
        cancel_order.type = OrderType.CANCEL
        cancel_order.ref = f'{order.ref}_cancel'
        if self.valid_order(cancel_order):
            self._pending_orders[order.market_id][order.side] = cancel_order
            self.queue_order(cancel_order)

    def valid_order(self, o, bypass_tracking=False):
        """
        Returns a boolean indicating if an order is valid (price, position, capital).
        """
        # 4c: check order validity
        # ensure that price is valid
        valid = self.markets[o.market_id]['minimum'] <= o.price <= self.markets[o.market_id]['maximum']
        # ensure no current pending orders for chosen order position, unless tracking bypass is requested
        valid &= bypass_tracking | (not (self._pending_orders.get(o.market_id, {}).get(o.side)))
        if o.type==OrderType.LIMIT:
            # ensure no current active orders as well when opening a new position, unless tracking bypass is requested
            valid &= bypass_tracking | (not (self._active_orders.get(o.market_id, {}).get(o.side)))
            # check if sufficient capital for limit orders
            if o.side==OrderSide.BUY:
                valid &= (self._holdings['cash']['available_cash'] >= o.price*o.units)
            elif o.side==OrderSide.SELL:
                valid &= (self._holdings['markets'][o.market_id]['available_units'] >= o.units)
        return valid

    def send_if_valid_order(self, order):
        """
        Sends a valid order using the primary order queue.
        """
        if self.valid_order(order):
            self._pending_orders[order.market_id][order.side] = order
            self.queue_order(order)

    def run(self):
        self.initialise()
        self.start()


if __name__ == "__main__":
    # 1a: trading account details
    FM_ACCOUNT = ''
    FM_EMAIL = ''
    FM_PASSWORD = ''
    MARKETPLACE_ID = 0

    bot = CAPMBot(FM_ACCOUNT, FM_EMAIL, FM_PASSWORD, MARKETPLACE_ID)
    bot.run()
