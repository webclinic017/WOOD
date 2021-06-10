import threading

from typing import Callable, List, Any
import pandas

import atexit
import logging
from datetime import datetime

import numpy as np

import calendar
import math
from enum import Enum

import pandas as pd
import numpy as np


class TableManagerListener(fxcorepy.AO2GTableManagerListener):
    """The class implements the abstract class AO2GTableManagerListener and calls the passed function when the table manager status changes."""
    def __init__(self, on_status_changed_callback: Callable[
        [fxcorepy.AO2GTableManagerListener, fxcorepy.O2GTableManager, fxcorepy.O2GTableManagerStatus], None] = None) \
            -> None:
        """ The constructor.
        
            Parameters
            ----------
            on_status_changed_callback : typing.Callable[[AO2GTableManagerListener, O2GTableManager, O2GTableManagerStatus], None]
                The function that is called when the table manager status changes.
        
            Returns
            -------
            None
        
        """
        fxcorepy.AO2GTableManagerListener.__init__(self)
        self._on_status_changed_callback = on_status_changed_callback
        self._status = fxcorepy.O2GTableManagerStatus.TABLES_LOADING
        self._semaphore = threading.Event()

    @property
    def status(self) -> fxcorepy.O2GTableManagerStatus:
        """ Gets the current status of the table manager.
        
            Returns
            -------
            O2GTableManagerStatus
        
        """
        return self._status

    def set_callback(self, on_status_changed_callback: Callable[
        [fxcorepy.AO2GTableManagerListener, fxcorepy.O2GTableManager, fxcorepy.O2GTableManagerStatus], None]) \
            -> None:
        """ Sets a callback function.
        
            Parameters
            ----------
            on_status_changed_callback : 
                The function that is called when the table manager status changes.
        
            Returns
            -------
            None
        
        """
        self._on_status_changed_callback = on_status_changed_callback

    def on_status_changed(self, status: fxcorepy.O2GTableManagerStatus, table_manager: fxcorepy.O2GTableManager) \
            -> None:  # native call
        """ Implements the method AO2GTableManagerListener.on_status_changed and calls the function that processes notifications about the table manager status change. The function is passed in the constructor or set by the set_callback method.
        
            Returns
            -------
            
        
        """
        self._status = status
        if self._on_status_changed_callback:
            self._on_status_changed_callback(self, table_manager, status)

        if status != fxcorepy.O2GTableManagerStatus.TABLES_LOADING:
            self._semaphore.set()

    def wait_event(self) -> bool:
        """Reserved for future use."""
        return self._semaphore.wait(30)

    def reset(self) -> None:
        """ Resets the flag that is set when the table manager status changes to TABLES_LOADED or TABLES_LOAD_FAILED.
        
            Returns
            -------
            None
        
        """
        self._status = fxcorepy.O2GTableManagerStatus.TABLES_LOADING
        self._semaphore.clear()


class Common:
    """The class contains helper functions."""
    @staticmethod
    def create_table_listener(table: fxcorepy.O2GTable = None,
                              on_add_callback: Callable[
                                  [fxcorepy.AO2GTableListener, str,
                                   fxcorepy.O2GRow], None
                              ] = None,
                              on_delete_callback: Callable[
                                  [fxcorepy.AO2GTableListener, str,
                                   fxcorepy.O2GRow], None
                              ] = None,
                              on_change_callback: Callable[
                                  [fxcorepy.AO2GTableListener, str,
                                   fxcorepy.O2GRow], None
                              ] = None,
                              on_status_change_callback: Callable[
                                  [fxcorepy.AO2GTableListener,
                                   fxcorepy.O2GTableStatus], None] = None) -> TableListener:
        """ Creates a table listener.
        
            Parameters
            ----------
            on_add_callback : typing.Callable[[AO2GTableListener, str, O2GRow], None]
                The function that is called when a row is added to the table.
            on_delete_callback : typing.Callable[[AO2GTableListener, str, O2GRow], None]
                The function that is called when a row is deleted from the table.
            on_change_callback : typing.Callable[[AO2GTableListener, str, O2GRow], None]
                The function that is called when a row in the table is changed.
            on_status_change_callback : typing.Callable[[AO2GTableListener, O2GTableStatus], None]
                The function that is called when the table status is changed.
        
            Returns
            -------
            TableListener
        
        """
        table_listener = TableListener(
            table, on_change_callback, on_add_callback, on_delete_callback, on_status_change_callback
        )

        return table_listener

    @staticmethod
    def subscribe_table_updates(table: fxcorepy.O2GTable = None,
                                on_add_callback: Callable[
                                    [fxcorepy.AO2GTableListener, str, fxcorepy.O2GRow], None] = None,
                                on_delete_callback: Callable[
                                    [fxcorepy.AO2GTableListener, str, fxcorepy.O2GRow], None] = None,
                                on_change_callback: Callable[
                                    [fxcorepy.AO2GTableListener, str, fxcorepy.O2GRow], None] = None,
                                on_status_change_callback: Callable[
                                    [fxcorepy.AO2GTableListener, fxcorepy.O2GTableStatus], None] = None) \
            -> TableListener:
        """ Creates a table listener and subscribes it to updates of a certain table.
        
            Parameters
            ----------
            table : O2GTable
                An instance of O2GTable.
            on_add_callback : typing.Callable[[AO2GTableListener, str, O2GRow], None]
                The function that is called when a row is added to the table.
            on_delete_callback : typing.Callable[[AO2GTableListener, str, O2GRow], None]
                The function that is called when a row is deleted from the table.
            on_change_callback : typing.Callable[[AO2GTableListener, str, O2GRow], None]
                The function that is called when a row in the table is changed.
            on_status_change_callback : typing.Callable[[AO2GTableListener, O2GTableStatus], None]
                The function that is called when the table status is changed.
        
            Returns
            -------
            TableListener
        
        """
        table_listener = Common.create_table_listener(table, on_add_callback, on_delete_callback,
                                                      on_change_callback, on_status_change_callback)
        table_listener.subscribe()
        return table_listener

    @staticmethod
    def join_to_new_group_request(fc: ForexConnect,
                                  account_id: str,
                                  primary_id: str,
                                  secondary_id: str,
                                  contingency_type: int) -> fxcorepy.O2GRequest:
        """ Creates a request for joining two specifed orders to a new contingency group.
        
            Parameters
            ----------
            fc : ForexConnect
                An instance of ForexConnect.
            account_id : str
                The unique identifier of the account the orders belong to.
            primary_id : str
                The unique identifier of the order that will be primary in the contingency group.
            secondary_id : str
                The unique identifier of the order that will be secondary in the contingency group.
            contingency_type : int
                The type of the contingency group to be created.
        
            Returns
            -------
            O2GRequest
        
        """
        request_params = {
            fxcorepy.O2GRequestParamsEnum.COMMAND: fxcorepy.Constants.Commands.JOIN_TO_NEW_CONTINGENCY_GROUP,
            fxcorepy.O2GRequestParamsEnum.CONTINGENCY_GROUP_TYPE: contingency_type,
            primary_id: {
                fxcorepy.O2GRequestParamsEnum.ORDER_ID: primary_id,
                fxcorepy.O2GRequestParamsEnum.ACCOUNT_ID: account_id},
            secondary_id: {
                fxcorepy.O2GRequestParamsEnum.ORDER_ID: secondary_id,
                fxcorepy.O2GRequestParamsEnum.ACCOUNT_ID: account_id}
        }
        return fc.create_request(request_params)

    @staticmethod
    def is_order_exist(fc: ForexConnect, account_id: str, order_id: str) -> bool:
        """ Checks whether an order for a certain instrument exists on a specific account.
        
            Parameters
            ----------
            fc : ForexConnect
                An instance of ForexConnect.
            account_id : str
                The unique identifier of the account.
            offer_id : str
                The unique identification number of the instrument.
        
            Returns
            -------
            bool
        
        """
        response_reader = Common.refresh_table_by_account(fc, fc.ORDERS, account_id)

        for order_row in response_reader:
            if order_id == order_row.order_id:
                return True
        return False

    @staticmethod
    def refresh_table_by_account(fc: ForexConnect,
                                 table: fxcorepy.O2GTable,
                                 account_id: str) -> Any:
        """ Refreshes a table for a certain account.
        
            Parameters
            ----------
            fc : ForexConnect
                An instance of ForexConnect.
            table : O2GTable
                An instance of O2GTable.
            account_id : str
                The account identifier.
        
            Returns
            -------
            O2GGenericTableResponseReader
        
        """
        request_factory = fc.session.request_factory

        if request_factory is None:
            raise Exception("Cannot create request factory")

        request = request_factory.create_refresh_table_request_by_account(table, account_id)

        if request is None:
            raise Exception(request_factory.last_error)

        return fc.send_request(request)

    @staticmethod
    def is_contingency_exist(fc: ForexConnect, account_id: str, contingency_id: str) -> bool:
        """ Checks whether a contingency group exists on a specific account.
        
            Parameters
            ----------
            fc : ForexConnect
                An instance of ForexConnect.
            account_id : str
                The unique identifier of the account.
            contingency_id : str
                The unique identifier of the contingency group.
        
            Returns
            -------
            bool
        
        """
        response_reader = Common.refresh_table_by_account(fc, fc.ORDERS, account_id)

        for order_row in response_reader:
            if contingency_id == order_row.contingent_order_id:
                return True
        return False

    @staticmethod
    def fill_request_ids(request_ids: List[str],
                         request: fxcorepy.O2GRequest) -> None:
        """Reserved for future use."""
        children_count = len(request)

        if children_count == 0:
            request_ids.append(request.request_id)
            return

        for i in range(children_count):
            Common.fill_request_ids(request_ids, request[i])

    @staticmethod
    def get_account(fc: ForexConnect, account_id=None, additional_check_func:  Callable[
        [fxcorepy.O2GAccountRow], bool] = None) \
            -> fxcorepy.O2GAccountRow:
        """ Gets an instance of O2GAccountRow by Account ID or searches for a trading account with no limitations on account operations and/or satisfying a certain condition.
        
            Parameters
            ----------
            fc : ForexConnect
                An instance of ForexConnect.
            account_id : str
                The unique identifier of the account.
            additional_check_func : typing.Callable[[O2GAccountRow], bool]
                The function that is called if a user specifies an additional check for the account.
        
            Returns
            -------
            O2GAccountRow
        
        """
        accounts_response_reader = fc.get_table_reader(
            fxcorepy.O2GTableType.ACCOUNTS)
        for account in accounts_response_reader:
            account_kind = account.account_kind
            if account_kind == "32" or account_kind == "36":
                if account.margin_call_flag == "N" and \
                        (account_id is None or account_id == account.account_id):
                    if additional_check_func is not None:
                        if additional_check_func(account):
                            return account
                    else:
                        return account
        return None

    @staticmethod
    def get_offer(fc: ForexConnect, instrument: str) -> fxcorepy.O2GOfferRow:
        """ Gets an instance of O2GOfferRow for a specified instrument.
        
            Parameters
            ----------
            fc : ForexConnect
                An instance of ForexConnect.
            instrument : str
                The name of the instrument.
        
            Returns
            -------
            O2GOfferRow
        
        """
        try:
            offers_table = fc.get_table(fxcorepy.O2GTableType.OFFERS)
            for offer in offers_table:
                if offer.instrument == instrument:
                    if offer.subscription_status == "T":
                        return offer
        except TableManagerError:
            offers_response_reader = fc.get_table_reader(
                fxcorepy.O2GTableType.OFFERS)
            for offer in offers_response_reader:
                if offer.instrument == instrument and offer.subscription_status == "T":
                    return offer
        return None

    @staticmethod
    def get_trade(fc: ForexConnect, account_id: str, offer_id: str) -> fxcorepy.O2GTradeTableRow:
        """ Gets an instance of O2GTradeRow from the Trades table by Account ID and Offer ID.
        
            Parameters
            ----------
            fc : ForexConnect
                An instance of ForexConnect.
            account_id : str
                The identifier of the account the position is opened on.
            offer_id : str
                The unique identification number of the instrument the position is opened in.
        
            Returns
            -------
            O2GTradeRow
        
        """
        try:
            trades_table = fc.get_table(fxcorepy.O2GTableType.TRADES)
            for trade in trades_table:
                if trade.account_id == account_id and trade.offer_id == offer_id:
                    return trade
        except TableManagerError:
            trades_response_reader = fc.get_table_reader(
                fxcorepy.O2GTableType.TRADES)
            for trade in trades_response_reader:
                if trade.account_id == account_id and trade.offer_id == offer_id:
                    return trade
        return None

    @staticmethod
    def _create_close_all_request_buy(account_id, offer_id):
        return {
            fxcorepy.O2GRequestParamsEnum.COMMAND: fxcorepy.Constants.Commands.CREATE_ORDER,
            fxcorepy.O2GRequestParamsEnum.NET_QUANTITY: "Y",
            fxcorepy.O2GRequestParamsEnum.ORDER_TYPE: fxcorepy.Constants.Orders.TRUE_MARKET_CLOSE,
            fxcorepy.O2GRequestParamsEnum.ACCOUNT_ID: account_id,
            fxcorepy.O2GRequestParamsEnum.OFFER_ID: offer_id,
            fxcorepy.O2GRequestParamsEnum.BUY_SELL: fxcorepy.Constants.BUY}

    @staticmethod
    def _create_close_all_request_sell(account_id, offer_id):
        return {
            fxcorepy.O2GRequestParamsEnum.COMMAND: fxcorepy.Constants.Commands.CREATE_ORDER,
            fxcorepy.O2GRequestParamsEnum.NET_QUANTITY: "Y",
            fxcorepy.O2GRequestParamsEnum.ORDER_TYPE: fxcorepy.Constants.Orders.TRUE_MARKET_CLOSE,
            fxcorepy.O2GRequestParamsEnum.ACCOUNT_ID: account_id,
            fxcorepy.O2GRequestParamsEnum.OFFER_ID: offer_id,
            fxcorepy.O2GRequestParamsEnum.BUY_SELL: fxcorepy.Constants.SELL}

    @staticmethod
    def create_close_trades_request(fc: ForexConnect, trade_ids: List[str] = None) -> fxcorepy.O2GRequest:
        """ Creates a request for closing a number of positions.
        
            Parameters
            ----------
            trade_ids : typing.List[str]
                The identifiers of the positions to be closed.
        
            Returns
            -------
            O2GRequest
        
        """
        request_params = {
            fxcorepy.O2GRequestParamsEnum.COMMAND: fxcorepy.Constants.Commands.CREATE_ORDER}
        trades_table = fc.get_table(ForexConnect.TRADES)
        order_info = {}
        for trade in trades_table:
            if trade_ids is None or trade.trade_id in trade_ids:
                offer_id = trade.offer_id
                buy_sell = trade.buy_sell
                account_id = trade.account_id
                s_key = "{0}_{1}".format(account_id, offer_id)
                if buy_sell == fxcorepy.Constants.BUY:
                    if s_key not in order_info or fxcorepy.Constants.BUY not in order_info[s_key]:
                        request_params[offer_id] = Common._create_close_all_request_sell(account_id, offer_id)
                        if s_key not in order_info:
                            order_info[s_key] = {}
                        order_info[s_key][fxcorepy.Constants.BUY] = True
                else:
                    if s_key not in order_info or fxcorepy.Constants.SELL not in order_info[s_key]:
                        request_params[offer_id] = Common._create_close_all_request_buy(account_id, offer_id)
                        if s_key not in order_info:
                            order_info[s_key] = {}
                        order_info[s_key][fxcorepy.Constants.SELL] = True
        return fc.create_request(request_params)

    @staticmethod
    def add_orders_into_group_request(fc: ForexConnect,
                                      account_id: str,
                                      contingency_id: str,
                                      order_ids: List[str],
                                      contingency_type: int) -> fxcorepy.O2GRequest:
        """ Creates a request for adding certain orders to an existing contingency group.
        
            Parameters
            ----------
            fc : ForexConnect
                An instance of ForexConnect.
            account_id : str
                The identifier of the account the orders belong to.
            contingency_id : str
                The identifier of an existing contingency group.
            order_ids : typing.List[str]
                The identifiers of the orders to be added to the contingency group.
            contingency_type : int
                The type of the contingency group.
        
            Returns
            -------
            O2GRequest
        
        """
        request_params = {
            fxcorepy.O2GRequestParamsEnum.COMMAND: fxcorepy.Constants.Commands.JOIN_TO_EXISTING_CONTINGENCY_GROUP,
            fxcorepy.O2GRequestParamsEnum.CONTINGENCY_GROUP_TYPE: contingency_type,
            fxcorepy.O2GRequestParamsEnum.CONTINGENCY_ID: contingency_id}

        for str_order_id in order_ids:
            child_request = {
                fxcorepy.O2GRequestParamsEnum.ORDER_ID: str_order_id,
                fxcorepy.O2GRequestParamsEnum.ACCOUNT_ID: account_id}
            request_params[str_order_id] = child_request

        return fc.create_request(request_params)

    @staticmethod
    def convert_table_to_dataframe(table: fxcorepy.O2GTable) -> pandas.DataFrame:
        """ Converts O2GTable to a pandas.DataFrame.
        
            Parameters
            ----------
            table : O2GTable
                An instance of O2GTable.
        
            Returns
            -------
            pandas.DataFrame
        
        """
        column_names = []
        for column in table.columns:
            if column.is_key:
                continue
            column_names.append(column.id)

        data = []
        index = []
        for row in table:
            key_value = row[table.columns.key_column.id]
            r = {}
            for column in column_names:
                r[column] = row[column]
            data.append(r)
            index.append(key_value)
        return pandas.DataFrame(data, index=index, columns=column_names).sort_index(axis=1).sort_index()

    @staticmethod
    def convert_row_to_dataframe(row: fxcorepy.O2GRow) -> pandas.DataFrame:
        """ Converts O2GRow to a pandas.DataFrame.
        
            Parameters
            ----------
            row : O2GRow
                An instance of O2GRow.
        
            Returns
            -------
            pandas.DataFrame
        
        """
        row_obj = {}
        column_names = []
        for column in row.columns:
            if column.is_key:
                continue
            row_obj[column.id] = row[column.id]
            column_names.append(column.id)
        return pandas.DataFrame([row_obj], index=[row[row.columns.key_column.id]],
                                columns=column_names).sort_index(axis=1)


class EachRowListener(fxcorepy.AO2GEachRowListener):
    """The class implements the abstract class AO2GEachRowListener and calls the passed function on the iteration through rows of a table."""
    def __init__(self, on_each_row_callback: Callable[[fxcorepy.AO2GEachRowListener, str, fxcorepy.O2GRow], None] = None,
                 data: Any = None) -> None:
        """ The constructor.
        
            Parameters
            ----------
            on_each_row_callback : 
                The function that is called on an iteration through rows of a table.
            data : typing.Any
                Any user's object. The default value of the parameter is None.
        
            Returns
            -------
            None
        
        """
        fxcorepy.AO2GEachRowListener.__init__(self)
        self._on_each_row_callback = on_each_row_callback
        self._data = data

    @property
    def data(self) -> Any:
        """ Gets data passed in the constructor.
        
            Returns
            -------
            typing.Any
        
        """
        return self._data

    def on_each_row(self, row_id: str, row_data: fxcorepy.O2GRow) -> None:  # native call
        """ Implements the method AO2GEachRowListener.on_each_row and calls the function that processes notifications on the iteration through the rows of a table. The function is passed in the constructor.
        
            Returns
            -------
            
        
        """
        if self._on_each_row_callback:
            self._on_each_row_callback(self, row_id, row_data)


class RequestFailedError(Exception):
    """Reserved for future use."""
    msg = ""

    def __init__(self, error_text):
        Exception.__init__(self)
        self.msg = error_text

    def __str__(self):
        return self.msg


class TableManagerError(Exception):
    """Reserved for future use."""
    msg = ""

    def __init__(self, error_text):
        Exception.__init__(self)
        self.msg = error_text

    def __str__(self):
        return self.msg


class LoginFailedError(Exception):
    """Reserved for future use."""
    msg = ""

    def __init__(self, error_text):
        Exception.__init__(self)
        self.msg = error_text

    def __str__(self):
        return self.msg


class TimeFrameError(Exception):
    """Reserved for future use."""
    msg = ""

    def __init__(self, error_text):
        Exception.__init__(self)
        self.msg = error_text

    def __str__(self):
        return self.msg


@atexit.register
def on_exit():
    fxcorepy.O2GTransport.finalize_wrapper()


class ForexConnect:
    """The class is intended for working with a session."""
    _SESSION = None
    _STATUS_LISTENER = None
    _TABLE_MANAGER_LISTENER = None
    _START_URL = "http://fxcorporate.com/Hosts.jsp"
    _LISTENER = None

    OFFERS = fxcorepy.O2GTableType.OFFERS
    ACCOUNTS = fxcorepy.O2GTableType.ACCOUNTS
    TRADES = fxcorepy.O2GTableType.TRADES
    CLOSED_TRADES = fxcorepy.O2GTableType.CLOSED_TRADES
    ORDERS = fxcorepy.O2GTableType.ORDERS
    MESSAGES = fxcorepy.O2GTableType.MESSAGES
    SUMMARY = fxcorepy.O2GTableType.SUMMARY

    class TableUpdateType(fxcorepy.O2GTableUpdateType):
        pass

    class ResponseType(fxcorepy.O2GResponseType):
        pass

    class TableManagerStatus(fxcorepy.O2GTableManagerStatus):
        pass

    def __init__(self) -> None:
        self._session = ForexConnect._SESSION
        self._status_listener = ForexConnect._STATUS_LISTENER
        self._table_manager_listener = ForexConnect._TABLE_MANAGER_LISTENER
        self._prev_response_listener = None
        self._async_response_listener = ResponseListenerAsync(self)
        self._async_response_listener_subscribed = False
        self._start_url = ForexConnect._START_URL
        self._session_id = None
        self._pin = None
        self._table_manager = None

    @property
    def session(self) -> fxcorepy.O2GSession:
        """ Gets an instance of the current session.
        
            Returns
            -------
            O2GSession
        
        """
        return self._session

    @property
    def table_manager(self) -> fxcorepy.O2GTableManager:
        """ Gets the current table manager of the session.
        
            Returns
            -------
            O2GTableManager
        
        """
        return self._table_manager

    @property
    def response_listener(self) -> ResponseListenerAsync:
        """Reserved for future use."""
        return self._async_response_listener

    @property
    def login_rules(self) -> fxcorepy.O2GLoginRules:
        """ Gets the rules used for the currently established session.
        
            Returns
            -------
            O2GLoginRules
        
        """
        login_rules = self._session.login_rules

        if login_rules is None:
            raise Exception("Cannot get login rules")

        return login_rules

    def __enter__(self):
        return self

    def __exit__(self, tp, value, traceback):
        self.logout()

    def _login(self,
               login_function,
               login_params,
               session_id: str = None,
               pin: str = None,
               session_status_callback: Callable[[fxcorepy.O2GSession,
                                                  fxcorepy.AO2GSessionStatus.O2GSessionStatus], None] = None,
               use_table_manager: bool = True,
               table_manager_callback: Callable[[fxcorepy.O2GTableStatus,
                                                 fxcorepy.O2GTableManager],
                                                None] = None) -> fxcorepy.O2GSession:

        self._session_id = session_id
        self._pin = pin
        self._session = fxcorepy.O2GTransport.create_session()

        if use_table_manager:
            self._table_manager_listener = TableManagerListener(table_manager_callback)
            self._session.use_table_manager(fxcorepy.O2GTableManagerMode.YES,
                                            self._table_manager_listener)

        self._status_listener = SessionStatusListener(self._session,
                                                      session_id,
                                                      pin,
                                                      session_status_callback)
        self._session.subscribe_session_status(self._status_listener)
        self._status_listener.reset()

        login_function = getattr(self._session, login_function)
        if not login_function(*login_params):
            raise LoginFailedError("The login method returned an exception."
                                   " This may be caused by the incorrect session status.")

        timeout = self._status_listener.wait_event()
        status_connect = self._status_listener.connected

        self._table_manager = self._session.table_manager

        if not timeout:
            raise LoginFailedError("Wait timeout exceeded")

        if not status_connect:
            if self._status_listener.last_error:
                raise LoginFailedError(self._status_listener.last_error)
            else:
                raise LoginFailedError("Not connected")

        if use_table_manager:
            self._table_manager_listener.wait_event()
            if self._table_manager.status != fxcorepy.O2GTableManagerStatus.TABLES_LOADED:
                raise LoginFailedError("Table manager not ready")

        return self._session

    def login(self,
              user_id: str,
              password: str,
              url: str = _START_URL,
              connection: str = "Demo",
              session_id: str = None,
              pin: str = None,
              session_status_callback: Callable[[fxcorepy.O2GSession,
                                                 fxcorepy.AO2GSessionStatus.O2GSessionStatus], None] = None,
              use_table_manager: bool = True,
              table_manager_callback: Callable[[fxcorepy.O2GTableStatus,
                                                fxcorepy.O2GTableManager],
                                               None] = None) -> fxcorepy.O2GSession:
        """ Creates a trading session and starts the connection with the specified trade server.
        
            Parameters
            ----------
            user_id : str
                The user name.
            password : str
                The password.
            url : str
                The URL of the server. The URL must be a full URL, including the path to the host descriptor.
            connection : str
                The name of the connection, for example Demo or Real.
            session_id : str
                The database name. Required only for users who have accounts in more than one database.
            pin : str
                The PIN code. Required only for users who have PIN codes.
            session_status_callback : typing.Callable[[O2GSession, AO2GSessionStatus.O2GSessionStatus], None]
                The function that is called when the session status changes.
            use_table_manager : bool
                Defines whether ForexConnect is started with the table manager.
            table_manager_callback : typing.Callable[[O2GTableManager, O2GTableStatus], None]
                The function that is called when the table manager status changes.
        
            Returns
            -------
            O2GSession
        
        """
        return self._login("login", (user_id, password, url, connection), session_id, pin, session_status_callback,
                           use_table_manager, table_manager_callback)

    def login_with_token(self,
                         user_id: str,
                         token: str,
                         url: str = _START_URL,
                         connection: str = "Demo",
                         session_id: str = None,
                         pin: str = None,
                         session_status_callback: Callable[[fxcorepy.O2GSession,
                                                            fxcorepy.AO2GSessionStatus.O2GSessionStatus], None] = None,
                         use_table_manager: bool = True,
                         table_manager_callback: Callable[[fxcorepy.O2GTableStatus,
                                                           fxcorepy.O2GTableManager],
                                                          None] = None) -> fxcorepy.O2GSession:
        """ Creates a second trading session and starts the connection with the specified trade server using a token.
        
            Parameters
            ----------
            user_id : str
                The user name.
            token : str
                The token.
            url : str
                The URL of the server. The URL must be a full URL, including the path to the host descriptor.
            connection : str
                The name of the connection, for example Demo or Real.
            session_id : str
                The database name. Required only for users who have accounts in more than one database.
            pin : str
                The PIN code. Required only for users who have PIN codes. The default value of the parameter is None.
            session_status_callback : typing.Callable[[O2GSession, AO2GSessionStatus.O2GSessionStatus], None]
                The function that is called when the session status changes. The default value of the parameter is None.
            use_table_manager : bool
                Defines whether ForexConnect is started with the table manager. The default value of the parameter is True.
            table_manager_callback : typing.Callable[[O2GTableStatus, O2GTableManager], None]
                The function that is called when the table manager status changes. The default value of the parameter is None
        
            Returns
            -------
            O2GSession
        
        """
        return self._login("login_with_token", (user_id, token, url, connection), session_id, pin,
                           session_status_callback, use_table_manager, table_manager_callback)

    def logout(self) -> None:
        self.unsubscribe_response()
        if self._session is None or \
                self._session.session_status == fxcorepy.AO2GSessionStatus.O2GSessionStatus.DISCONNECTED:
            return
        """ Closes the trading session and connection with the trade server.
        
            Returns
            -------
            None
        
        """
        self._status_listener.reset()
        self._session.logout()
        self._status_listener.wait_event()
        self._session.unsubscribe_session_status(self._status_listener)

        self._session = None
        self._status_listener = None
        self._table_manager = None

    def set_session_status_listener(self, listener: Callable[
        [fxcorepy.O2GSession, fxcorepy.AO2GSessionStatus.O2GSessionStatus], None]) \
            -> None:
        """ Sets a session status listener on login or sets a new session status listener when necessary.
        
            Parameters
            ----------
            listener : typing.Callable[[O2GSession, AO2GSessionStatus.O2GSessionStatus], None]
                The function that is called when the session status changes.
        
            Returns
            -------
            None
        
        """
        self._status_listener.set_callback(listener)

    def get_table_reader(self, table_type: fxcorepy.O2GTable,
                         response: fxcorepy.O2GResponse = None) -> fxcorepy.O2GGenericTableResponseReader:
        """ Gets an instance of a table reader.
        
            Parameters
            ----------
            table_type : O2GTableType
                The identifier of the table (see O2GTableType).
            response : O2GResponse
                An instance of O2GResponse to get a reader for.
        
            Returns
            -------
            typing.object
        
        """
        if response is None:
            login_rules = self._session.login_rules
            response = login_rules.get_table_refresh_response(table_type)

        return self.create_reader(response)

    def get_table(self, table_type: fxcorepy.O2GTable) -> fxcorepy.O2GTable:
        """ Gets a specified table.
        
            Parameters
            ----------
            table_type : O2GTableType
                The identifier of the table. For a complete list of tables, see O2GTableType.
        
            Returns
            -------
            O2GTable
        
        """
        if self._table_manager_listener is None:
            raise TableManagerError(
                'Need login with flag "useTableManager"')

        if not self._table_manager_listener.wait_event():
            raise TableManagerError("Wait timeout exceeded")

        if self._table_manager_listener.status == fxcorepy.O2GTableManagerStatus.TABLES_LOAD_FAILED:
            raise TableManagerError(
                "Cannot refresh all tables of table manager")

        return self._session.table_manager.get_table(table_type)

    def subscribe_response(self) -> None:
        """Reserved for future use."""
        if self._async_response_listener_subscribed:
            return
        self._async_response_listener_subscribed = True
        self._session.subscribe_response(self._async_response_listener)

    def unsubscribe_response(self) -> None:
        """Reserved for future use."""
        if not self._async_response_listener_subscribed:
            return
        self._async_response_listener_subscribed = False
        self._session.unsubscribe_response(self._async_response_listener)

    def send_request_async(self,
                           request: fxcorepy.O2GRequest,
                           listener: ResponseListener = None):
        """ Sends a request.
        
            Parameters
            ----------
            request : O2GRequest
                An instance of O2GRequest.
            listener : ResponseListener
                An instance of ResponseListener. The default value of the parameter is None.
        
            Returns
            -------
            None
        
        """
        if listener is None:
            listener = ResponseListener(self._session)
        self.subscribe_response()
        request_ids = []
        Common.fill_request_ids(request_ids, request)
        listener.set_request_ids(request_ids)
        self._async_response_listener.add_response_listener(listener)
        self._session.send_request(request)

    def send_request(self,
                     request: fxcorepy.O2GRequest,
                     listener: ResponseListener = None) -> Any:
        """ Sends a request and returns the appropriate response reader or a bool value.
        
            Parameters
            ----------
            request : O2GRequest
                An instance of O2GRequest.
            listener : ResponseListener
                An instance of ResponseListener. The default value of the parameter is None.
        
            Returns
            -------
            typing.Any
        
        """
        if threading.current_thread() != threading.main_thread():
            logging.warning("Calling the send_request method is not from the main thread. "
                            "If you call the send_request method from a callback, you can get the application freezed. "
                            "It is recommended to use send_request_async method.")
        if listener is None:
            listener = ResponseListener(self._session)
        listener.reset()
        self.send_request_async(request, listener)

        if not listener.wait_event():
            self._async_response_listener.remove_response_listener(listener)
            raise RequestFailedError("Wait timeout exceeded")

        self._async_response_listener.remove_response_listener(listener)

        error = listener.error

        if error is None:
            return self.create_reader(listener.response)

        raise RequestFailedError(error)

    def get_history(self,
                    instrument: str,
                    timeframe: str,
                    date_from: datetime = None,
                    date_to: datetime = None,
                    quotes_count: int = -1,
                    candle_open_price_mode=fxcorepy.O2GCandleOpenPriceMode.PREVIOUS_CLOSE
                    ) -> np.ndarray:
        """ Gets price history of a certain instrument with a certain timeframe for a specified period or a certain number of bars/ticks.
        
            Parameters
            ----------
            instrument : str
                The symbol of the instrument. The instrument must be one of the instruments the ForexConnect session is subscribed to.
            timeframe : str
                The unique identifier of the timeframe. For details, see What is Timeframe?.
            date_from : datetime.datetime
                The date/time of the oldest bar/tick in the history.
            date_to : datetime.datetime
                The date/time of the most recent bar/tick in the history.
            quotes_count : int
                The number of bars/ticks in the history. The parameter is optional.
            candle_open_price_mode : 
                Constant representing the candles open price mode that indicates how the open price is determined.
        
            Returns
            -------
            numpy.ndarray
        
        """
        com = fxcorepy.PriceHistoryCommunicatorFactory.create_communicator(
            self._session, "./History")

        timeframe = com.timeframe_factory.create(timeframe)
        if not timeframe:
            raise TimeFrameError("Timeframe is incorrect")

        while not com.is_ready:
            pass

        com.candle_open_price_mode = candle_open_price_mode

        reader = com.get_history(instrument, timeframe, date_from, date_to, quotes_count)
        if timeframe.unit == fxcorepy.O2GTimeFrameUnit.TICK:
            result = np.zeros(len(reader), np.dtype([('Date', "M8[ns]"), ('Bid', "f8"), ("Ask", "f8")]))
            idx = -1
            for row in reader:
                idx += 1
                result[idx]['Date'] = np.datetime64(row.date)
                result[idx]['Bid'] = row.bid
                result[idx]['Ask'] = row.ask
            return result
        else:
            result = np.zeros(len(reader), np.dtype([
                ('Date', "M8[ns]"),
                ('BidOpen', 'f8'), ('BidHigh', 'f8'), ('BidLow', 'f8'), ('BidClose', 'f8'),
                ('AskOpen', 'f8'), ('AskHigh', 'f8'), ('AskLow', 'f8'), ('AskClose', 'f8'),
                ('Volume', 'i4')]))
            idx = -1
            for row in reader:
                idx += 1
                result[idx]['Date'] = np.datetime64(row.date)
                result[idx]['BidOpen'] = row.bid_open
                result[idx]['BidHigh'] = row.bid_high
                result[idx]['BidLow'] = row.bid_low
                result[idx]['BidClose'] = row.bid_close
                result[idx]['AskOpen'] = row.ask_open
                result[idx]['AskHigh'] = row.ask_high
                result[idx]['AskLow'] = row.ask_low
                result[idx]['AskClose'] = row.ask_close
                result[idx]['Volume'] = row.volume
            return result

    def create_reader(self,
                      response: fxcorepy.O2GResponse) -> Any:
        """ Creates a reader for a certain response using the method O2GResponseReaderFactory.create_reader.
        
            Parameters
            ----------
            response : O2GResponse
                An instance of O2GResponse.
        
            Returns
            -------
            typing.Any
        
        """
        factory = self._session.response_reader_factory
        if factory is None:
            raise Exception("Create ResponseReaderFactory failed")
        return factory.create_reader(response)

    def create_request(self, params: Dict[fxcorepy.O2GRequestParamsEnum, str],
                       request_factory: fxcorepy.O2GRequestFactory = None,
                       root: bool = True) -> fxcorepy.O2GRequest:
        """ Creates a request.
        
            Parameters
            ----------
            params : typing.Dict[O2GRequestParamsEnum, str]
                The request parameters. See O2GRequestParamsEnum.
            request_factory : O2GRequestFactory
                An instance of O2GRequestFactory.
            root : bool
                Defines whether the request is root. The default value of the parameter is True.
        
            Returns
            -------
            O2GRequest
        
        """
        if request_factory is None:
            request_factory = self._session.request_factory

            if request_factory is None:
                raise Exception("Can not create request factory")

        value_map = request_factory.create_value_map()
        items = params.items()

        for k, v in items:
            if isinstance(v, dict):
                value_map.append_child(
                    self.create_request(v, request_factory, False))

            else:
                if isinstance(v, str):
                    value_map.set_string(k, v)

                if isinstance(v, int):
                    value_map.set_int(k, v)

                if isinstance(v, bool):
                    value_map.set_bool(k, v)

                if isinstance(v, float):
                    value_map.set_double(k, v)

        if root:
            request = request_factory.create_order_request(value_map)
            if request is None:
                raise Exception(request_factory.last_error)
            return request

        else:
            return value_map

    def create_order_request(self, order_type: str,
                             command: fxcorepy.Constants.Commands = fxcorepy.Constants.Commands.CREATE_ORDER,
                             **kwargs: str) -> fxcorepy.O2GRequest:
        """ Creates a request for creating an order of a specified type using specified parameters.
        
            Parameters
            ----------
            order_type : str
                The type of the order. See Contansts.Orders.
            command : Commands
                The command. The default value of the parameter is fxcorepy.Constants.Commands.CreateOrder.
            str : kwargs
                The order parameters. For a full list of possible request parameters, see O2GRequestParamsEnum.
        
            Returns
            -------
            O2GRequest
        
        """
        if command is None:
            command = fxcorepy.Constants.Commands.CREATE_ORDER

        params = {
            fxcorepy.O2GRequestParamsEnum.COMMAND: command,
            fxcorepy.O2GRequestParamsEnum.ORDER_TYPE: order_type
        }

        for param in kwargs:
            enum = fxcorepy.O2GRequestParamsEnum.names[param]
            params[enum] = kwargs[param]

        return self.create_request(params)

    def get_timeframe(self, str_timeframe: str) -> fxcorepy.O2GTimeFrame:
        """ Gets an instance of a timeframe.
        
            Parameters
            ----------
            str_timeframe : str
                The unique identifier of the timeframe.
        
            Returns
            -------
            O2GTimeframe
        
        """
        return self._session.request_factory.timeframe_collection.get_by_id(
            str_timeframe)

    @staticmethod
    def parse_timeframe(timeframe: str) -> tuple:
        """ Parses a timeframe into O2GTimeframeUnit and the number of units.
        
            Parameters
            ----------
            timeframe : str
                The unique identifier of the timeframe.
        
            Returns
            -------
            O2GTimeframeUnit, int
        
        """
        if (len(timeframe) < 2):
            raise TimeFrameError("Timeframe is incorrect")

        available_units = {
            't': fxcorepy.O2GTimeFrameUnit.TICK,
            'm': fxcorepy.O2GTimeFrameUnit.MIN,
            'H': fxcorepy.O2GTimeFrameUnit.HOUR,
            'D': fxcorepy.O2GTimeFrameUnit.DAY,
            'W': fxcorepy.O2GTimeFrameUnit.WEEK,
            'M': fxcorepy.O2GTimeFrameUnit.MONTH,
        }

        try:
            unit = available_units[timeframe[0]]
        except KeyError as e:
            raise TimeFrameError("Unit is incorrect")

        try:
            size = int(timeframe[1:])
        except ValueError:
            raise TimeFrameError("Size must be a number")

        return unit, size



class LiveHistoryCreator:
    """The class is intended for updating the price history with live prices."""
    def __init__(self,
                 timeframe,
                 history=None,
                 limit=300,
                 candle_open_price_mode=fxcorepy.O2GCandleOpenPriceMode.PREVIOUS_CLOSE):
        """ The constructor.
        
            Parameters
            ----------
            timeframe : 
                The price history timeframe.
            history : numpy.ndarray
                The price history obtained using the method ForexConnect.get_history.
            limit : 
                The maximum number of bars/ticks in the price history. The default value of the parameter is 300.
            candle_open_price_mode : 
                Constant representing the candles open price mode that indicates how the open price is determined.
        
            Returns
            -------
            None
        
        """
        self.buffer = []
        self.limit = limit
        self._listeners = []
        self.history_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.last_ask_price = 0
        self.last_bid_price = 0
        self.last_volume = 0
        self._history = None
        self.history = history
        self.timeframe_unit, self.timeframe_size = ForexConnect.parse_timeframe(timeframe)
        self._candle_open_price_mode = candle_open_price_mode

    @property
    def candle_open_price_mode(self) -> fxcorepy.O2GCandleOpenPriceMode:
        return self._candle_open_price_mode

    @property
    def history(self):
        """ Gets or sets the updated price history.
        
            Returns
            -------
            pandas.DataFrame
        
        """
        return self._history

    @history.setter
    def history(self, history):
        if history is None:
            self._history = None
            return
        with self.history_lock:
            if isinstance(history, pd.DataFrame):
                self._history = history
            elif isinstance(history, np.ndarray):
                self._history = pd.DataFrame(data=history)
                self._history.set_index('Date', inplace=True)
            else:
                raise TypeError("Living history creator accept as history only object of type pandas. "
                                "DataFrame of numpy.ndarray. Received {item_type}".format(
                                         item_type=type(history)))
            if len(self._history.index) > self.limit:
                self._history = self._history.tail(self.limit)
            last_row = self._history.tail(1)
            if self.timeframe_unit != fxcorepy.O2GTimeFrameUnit.TICK:
                self.last_ask_price = last_row.AskClose.item()
                self.last_bid_price = last_row.BidClose.item()
                self.last_volume = last_row.Volume.item()

            while self._process_buffer(last_row):
                last_row = self._history.tail(1)

    def _process_buffer(self, last_row) -> bool:
        with self.buffer_lock:
            buffer = self.buffer.copy()
            self.buffer.clear()

        any_processed = False
        date = last_row.index
        for row in buffer:
            if date > row['Date']:
                continue
            self._add_or_update_internal(row)
            date = row['Date']
            any_processed = True
        return any_processed

    def add_or_update(self, row):
        """ Adds or updates ticks/bars.
        
            Parameters
            ----------
            row : O2GOfferRow
                An instance of O2GOfferRow.
        
            Returns
            -------
            None
        
        """
        dict_row = LiveHistoryCreator._convert_to_history_item(row)
        self.add_or_update_dict(dict_row)

    def add_or_update_dict(self, dict_row):
        """Reserved for future use."""
        if self._history is None:
            with self.buffer_lock:
                self.buffer.append(dict_row)
            return
        if not self.history_lock.acquire(True, 100):
            with self.buffer_lock:
                self.buffer.append(dict_row)
            return
        self._add_or_update_internal(dict_row)
        self.history_lock.release()

    def subscribe(self, on_add_bar_callback: Callable[
                     [pd.DataFrame], None
                 ]):
        """ Subscribes the function that is called when a bar/tick is added to or updated in the price history.
        
            Parameters
            ----------
            on_add_bar_callback : typing.Callable[[pandas.DataFrame], None]
                The function that is called when a bar/tick is added to or updated in the price history.
        
            Returns
            -------
            None
        
        """
        self._listeners.append(on_add_bar_callback)

    def unsubscribe(self, on_add_bar_callback: Callable[
                     [pd.DataFrame], None
                 ]):
        """ Unsubscribes the function that is called when a bar/tick is added to or updated in the price history.
        
            Parameters
            ----------
            on_add_bar_callback : typing.Callable[[pandas.DataFrame], None]
                The function that is called when a bar/tick is added to or updated in the price history.
        
            Returns
            -------
            None
        
        """
        try:
            self._listeners.remove(on_add_bar_callback)
        except ValueError:
            pass

    def _add_or_update_internal(self, row):
        current_unit = self.timeframe_unit
        if current_unit == fxcorepy.O2GTimeFrameUnit.TICK:
            item = pd.DataFrame([{'Date': row['Date'],
                                  'Bid': row['Bid'],
                                  'Ask': row['Ask']
                                  }]).set_index('Date')
            self._history = self._history.append(item, sort=True)
        else:
            timeframe = self._get_timeframe_by_time(row['Date'])
            if timeframe in self._history.index:
                last_row = self._history.loc[timeframe]
                if last_row.BidHigh.item() < row['Bid']:
                    self._history.at[timeframe, 'BidHigh'] = row['Bid']
                if last_row.AskHigh.item() < row['Ask']:
                    self._history.at[timeframe, 'AskHigh'] = row['Ask']
                if last_row.BidLow.item() > row['Bid']:
                    self._history.at[timeframe, 'BidLow'] = row['Bid']
                if last_row.AskLow.item() > row['Ask']:
                    self._history.at[timeframe, 'AskLow'] = row['Ask']
                self._history.at[timeframe, 'BidClose'] = row['Bid']
                self._history.at[timeframe, 'AskClose'] = row['Ask']
                if row['Volume'] > self.last_volume:
                    self._history.at[timeframe, 'Volume'] = last_row.Volume.item() + row['Volume'] - self.last_volume

            else:
                if len(self._history.index) == self.limit:
                    self._history = self._history.iloc[1:]

                ask_open = self.last_ask_price
                bid_open = self.last_bid_price

                if self._candle_open_price_mode == fxcorepy.O2GCandleOpenPriceMode.FIRST_TICK:
                    ask_open = row['Ask']
                    bid_open = row['Bid']

                item = pd.DataFrame([{'Date': np.datetime64(timeframe),
                                      'BidOpen': bid_open,
                                      'BidHigh': row['Bid'],
                                      'BidLow': row['Bid'],
                                      'BidClose': row['Bid'],
                                      'AskOpen': ask_open,
                                      'AskHigh': row['Ask'],
                                      'AskLow': row['Ask'],
                                      'AskClose': row['Ask'],
                                      'Volume': row['Volume'],
                                      }]).set_index('Date')
                self._history = self._history.append(item, sort=True)
                for callback in self._listeners:
                    callback(self._history)
            self.last_ask_price = row['Ask']
            self.last_bid_price = row['Bid']
            self.last_volume = row['Volume']

    @staticmethod
    def _convert_to_history_item(row):

        return {'Date': pd.to_datetime(row.time),
                'Bid': row.bid,
                'Ask': row.ask,
                'Volume': row.volume
                }

    def _get_timeframe_by_time(self, dt: datetime.datetime):
        current_unit = self.timeframe_unit
        current_size = self.timeframe_size

        step = 0
        if current_unit == fxcorepy.O2GTimeFrameUnit.MIN:
            step = datetime.timedelta(minutes=current_size)

        elif current_unit == fxcorepy.O2GTimeFrameUnit.HOUR:
            step = datetime.timedelta(hours=current_size)

        elif current_unit == fxcorepy.O2GTimeFrameUnit.DAY:
            step = datetime.timedelta(days=current_size)

        elif current_unit == fxcorepy.O2GTimeFrameUnit.WEEK:
            step = datetime.timedelta(weeks=current_size)

        elif current_unit == fxcorepy.O2GTimeFrameUnit.MONTH:
            month = (round(dt.month - 1 / current_size) * current_size) + 1
            return dt.replace(month=month, day=1, hour=0, minute=0, second=0, microsecond=0)

        step = step.total_seconds()  # step in seconds
        timespamp = dt.timestamp()
        new_timespamp = math.floor(timespamp / step) * step
        return datetime.datetime.utcfromtimestamp(new_timespamp)


class ResponseListener(fxcorepy.AO2GResponseListener):
    """The class implements the AO2GResponseListener abstract class and calls the passed functions on request completions/failures and table updates."""
    def __init__(self,
                 session: fxcorepy.O2GSession,
                 on_request_completed_callback: Callable[[str, fxcorepy.O2GResponse], bool] = None,
                 on_request_failed_callback: Callable[[str, str], bool] = None,
                 on_tables_updates_callback: Callable[[fxcorepy.O2GResponse], None] = None) -> None:
        """ The constructor.
        
            Parameters
            ----------
            session : O2GSession
                An instance of O2GSession.
            on_request_completed_callback : typing.Callable[[str, O2GResponse], bool]
                The function that is called when a notification about a request completion is received.
            on_request_failed_callback : typing.Callable[[str, str], bool]
                The function that is called when a notification about a request failure is received.
            on_tables_updates_callback : typing.Callable[O2GResponse, None]
                The function that is called when a notification about a table update is received.
        
            Returns
            -------
            None
        
        """
        super(ResponseListener, self).__init__()
        self._on_request_completed_callback = on_request_completed_callback
        self._on_request_failed_callback = on_request_failed_callback
        self._on_tables_updates_callback = on_tables_updates_callback
        self._response = None
        self._error = None
        self._event = threading.Event()
        self._request_ids = None
        self._session = session

    @property
    def is_set(self) -> bool:
        """Reserved for future use."""
        return self._event.is_set()

    @property
    def response(self) -> fxcorepy.O2GResponse:
        """ Gets an instance of O2GResponse if it has been received.
        
            Returns
            -------
            O2GResponse
        
        """
        return self._response

    @property
    def error(self) -> str:
        """ Gets a string representation of an error that occurred when processing a request.
        
            Returns
            -------
            str
        
        """
        return self._error

    @property
    def session(self) -> fxcorepy.O2GSession:
        """ Gets a string representation of an error that occurred when processing a request.
        
            Returns
            -------
            str
        
        """
        return self._session

    def set_request_id(self, request_id: str) -> None:
        """Reserved for future use."""
        self.set_request_ids([request_id])

    def on_request_completed(self,
                             request: str,
                             response: fxcorepy.O2GResponse) -> None:  # native call
        """ Implements the method AO2GEachRowListener.on_request_completed and calls the function that processes notifications about the successful request completion. The function is passed in the constructor.
        
            Returns
            -------
            
        
        """
        logging.debug("Request completed %s", request)
        if request in self._request_ids:
            self._response = response
            result = None
            if self._on_request_completed_callback:
                result = self._on_request_completed_callback(request, response)
            self._request_ids.remove(request)
            if not self._request_ids and (result is None or (isinstance(result, bool) and result)):
                self.stop_waiting()

    def on_request_failed(self, request: str,
                          error: str) -> None:  # native call
        """Reserved for future use."""
        logging.error("Request failed %s: %s", request, error)
        if request in self._request_ids:
            self._error = error
            self._request_ids.remove(request)
            result = None
            if self._on_request_failed_callback:
                result = self._on_request_failed_callback(request, error)

            if not self._request_ids and (result is None or (isinstance(result, bool) and result)):
                self.stop_waiting()

    def on_tables_updates(self, response: fxcorepy.O2GResponse) -> None:  # native call
        """ Implements the method AO2GEachRowListener.on_tables_updates and calls the function that processes notifications about table updates. The function is passed in the constructor.
        
            Returns
            -------
            
        
        """
        if self._on_tables_updates_callback:
            self._on_tables_updates_callback(response)

    def set_request_ids(self, request_ids: List[str]) -> None:
        """Reserved for future use."""
        self._request_ids = request_ids
        self.reset()

    def wait_event(self) -> bool:
        """Reserved for future use."""
        return self._event.wait(30)

    def stop_waiting(self) -> None:
        """ Stops waiting for a response.
        
            Returns
            -------
            None
        
        """
        self._event.set()

    def reset(self) -> None:
        """ Resets the response listener after a response is received.
        
            Returns
            -------
            None
        
        """
        self._error = None
        self._response = None
        self._event.clear()


class ResponseListenerAsync(fxcorepy.AO2GResponseListener):
    """"""
    def __init__(self, fc) -> None:
        """Reserved for future use."""
        super(ResponseListenerAsync, self).__init__()
        self._another_listener = []
        self._fc = fc

    def add_response_listener(self, another_listener: ResponseListener) -> None:
        """ 
        
            Parameters
            ----------
            another_listener : ResponseListener
                
        
            Returns
            -------
            None
        
        """

        if another_listener not in self._another_listener:
            self._another_listener.append(another_listener)

    def remove_response_listener(self, another_listener: ResponseListener):
        """ 
        
            Parameters
            ----------
            another_listener : ResponseListener
                
        
            Returns
            -------
            None
        
        """
        if another_listener not in self._another_listener:
            return
        self._another_listener.remove(another_listener)

    def on_request_completed(self,
                             request: str,
                             response: fxcorepy.O2GResponse) -> None:  # native call
        """Reserved for future use."""
        if len(self._another_listener) == 0:
            return
        for listener in self._another_listener:
            listener.on_request_completed(request, response)
            if listener.is_set:
                self.remove_response_listener(listener)

    def on_request_failed(self, request: str,
                          error: str) -> None:  # native call
        """Reserved for future use."""
        if len(self._another_listener) == 0:
            return
        for listener in self._another_listener:
            listener.on_request_failed(request, error)
            if listener.is_set:
                self.remove_response_listener(listener)

    def on_tables_updates(self, response: fxcorepy.O2GResponse) -> None:  # native call
        """Reserved for future use."""
        if len(self._another_listener) == 0:
            return
        for listener in self._another_listener:
            listener.on_tables_updates(response)
            if listener.is_set:
                self.remove_response_listener(listener)


class SessionStatusListener(fxcorepy.AO2GSessionStatus):
    """The class implements the abstract class AO2GSessionStatus and calls the passed function when the session status changes."""
    def __init__(self, session: fxcorepy.O2GSession, session_id: str, pin: str,
                 on_status_changed_callback: Callable[[fxcorepy.O2GSession, fxcorepy.AO2GSessionStatus.O2GSessionStatus], None] = None)\
            -> None:
        """ The constructor.
        
            Parameters
            ----------
            session : O2GSession
                An instance of O2GSession the session status listener listens to.
            session_id : str
                The identifier of the trading session. Must be one of the values returned by the
            pin : str
                The PIN code for the connection. If no PIN is required, specify an empty string ("").
            on_session_status_changed_callback : typing.Callable[[O2GSession, AO2GSessionStatus.O2GSessionStatus], None]
                The function that is called when the session status changes.
        
            Returns
            -------
            None
        
        """
        fxcorepy.AO2GSessionStatus.__init__(self)
        self.__session = session
        self.__session_id = session_id
        self.__pin = pin
        self.__semaphore = threading.Event()
        self.set_callback(on_status_changed_callback)
        self.__connected = False
        self.__disconnected = False
        self.__last_error = None

    def wait_event(self) -> bool:
        """Reserved for future use."""
        return self.__semaphore.wait(30)

    @property
    def connected(self) -> bool:
        """ Indicates if the session status is CONNECTED.
        
            Returns
            -------
            bool
        
        """
        return self.__connected

    @property
    def disconnected(self) -> bool:
        """ Indicates if the session status is DISCONNECTED.
        
            Returns
            -------
            bool
        
        """
        return self.__disconnected

    @property
    def last_error(self) -> str:
        """ Gets the last error received by the method on_login_failed.
        
            Returns
            -------
            str
        
        """
        return self.__last_error

    def reset(self) -> None:
        """ Resets the flag that is set when the session status changes to CONNECTED or DISCONNECTED.
        
            Returns
            -------
            None
        
        """
        self.__last_error = None
        self.__semaphore.clear()

    def set_callback(self, on_status_changed_callback: Callable[[fxcorepy.O2GSession,
                                                       fxcorepy.AO2GSessionStatus.O2GSessionStatus], None]) -> None:
        """ Sets a callback function.
        
            Parameters
            ----------
            on_session_status_changed_callback : typing.Callable[[O2GSession, AO2GSessionStatus.O2GSessionStatus], None]
                The function that is called when the session status changes.
        
            Returns
            -------
            None
        
        """
        self.__on_status_changed_callback = on_status_changed_callback

    def on_session_status_changed(self, status: fxcorepy.AO2GSessionStatus.O2GSessionStatus) -> None:  # native call
        """ Implements the method AO2GSessionStatus.on_session_status_changed and calls the function that processes notifications about the session status change. The function is passed in the constructor or set by the method set_callback.
        
            Returns
            -------
            
        
        """
        self.__connected = status == fxcorepy.AO2GSessionStatus.O2GSessionStatus.CONNECTED
        self.__disconnected = status == fxcorepy.AO2GSessionStatus.O2GSessionStatus.DISCONNECTED

        if self.__on_status_changed_callback is not None:
            self.__on_status_changed_callback(self.__session, status)

        if status == fxcorepy.AO2GSessionStatus.O2GSessionStatus.CONNECTED or \
                status == fxcorepy.AO2GSessionStatus.O2GSessionStatus.DISCONNECTED:
            self.__semaphore.set()

        if status == fxcorepy.AO2GSessionStatus.O2GSessionStatus.TRADING_SESSION_REQUESTED:
            if self.__session_id != "":
                self.__session.set_trading_session(self.__session_id, self.__pin)

    def on_login_failed(self, err: str) -> None:
        """ Implements the method AO2GSessionStatus.on_login_failed and saves the error in case of a failed login.
        
            Returns
            -------
            
        
        """
        self.__last_error = err


class TableListener(fxcorepy.AO2GTableListener):
    """The class implements the abstract class AO2GTableListener and calls the passed functions on the appropriate trading tables events: adding/changing/deleting of rows and table status changes."""
    def __init__(self,
                 table: fxcorepy.O2GTable = None,
                 on_changed_callback: Callable[
                     [fxcorepy.AO2GTableListener, str, fxcorepy.O2GRow], None
                 ] = None,
                 on_added_callback: Callable[
                     [fxcorepy.AO2GTableListener, str, fxcorepy.O2GRow], None
                 ] = None,
                 on_deleted_callback: Callable[
                     [fxcorepy.AO2GTableListener, str, fxcorepy.O2GRow], None
                 ] = None,
                 on_status_changed_callback: Callable[
                     [fxcorepy.AO2GTableListener,
                      fxcorepy.O2GTableStatus], None] = None
                 ) -> None:
        """ The constructor.
        
            Parameters
            ----------
            table : 
                An instance of O2GTable.
            on_added_callback : O2GTable)= None, (typing.Callable[[AO2GTableListener, str, O2GRow], None]
                The function that is called when a row is added to the table.
            on_deleted_callback : typing.Callable[[AO2GTableListener, str, O2GRow], None]
                The function that is called when a row is deleted from the table.
            on_changed_callback : typing.Callable[[AO2GTableListener, str, O2GRow], None]
                The function that is called when a row in the table is changed.
            on_status_changed_callback : typing.Callable[[AO2GTableListener, O2GTableStatus], None]
                The function that is called when a table status is changed.
        
            Returns
            -------
            None
        
        """
        fxcorepy.AO2GTableListener.__init__(self)
        self._on_changed_callback = on_changed_callback
        self._on_added_callback = on_added_callback
        self._on_deleted_callback = on_deleted_callback
        self._on_status_changed_callback = on_status_changed_callback
        self._table = table

    def on_added(self, row_id: str, row: fxcorepy.O2GRow) -> None:  # native call
        """ Implements the method AO2GTableListener.on_added and calls the function that processes notifications about the row addition to a table. The function is passed in the constructor.
        
            Returns
            -------
            
        
        """
        if self._on_added_callback:
            self._on_added_callback(self, row_id, row)

    def on_changed(self, row_id: str, row: fxcorepy.O2GRow) -> None:  # native call
        """ Implements the method AO2GTableListener.on_changed and calls the function that processes notifications about the row change in a table. The function is passed in the constructor.
        
            Returns
            -------
            
        
        """
        if self._on_changed_callback:
            self._on_changed_callback(self, row_id, row)

    def on_deleted(self, row_id: str, row: fxcorepy.O2GRow) -> None:  # native call
        """ Implements the method AO2GTableListener.on_deleted and calls the function that processes notifications about the row deletion from a table. The function is passed in the constructor.
        
            Returns
            -------
            
        
        """
        if self._on_deleted_callback:
            self._on_deleted_callback(self, row_id, row)

    def on_status_changed(self, status: fxcorepy.O2GTableStatus) -> None:  # native call
        """ Implements the method AO2GTableListener.on_status_changed and calls the function that processes notifications about the table status change. The function is passed in the constructor.
        
            Returns
            -------
            
        
        """
        if self._on_status_changed_callback:
            self._on_status_changed_callback(self, status)

    def _unsubscribe(self, type_u):
        self._table.unsubscribe_update(type_u, self)

    def unsubscribe(self) -> None:
        """ Unsubscribes a table listener from table updates.
        
            Returns
            -------
            None
        
        """
        if self._table is not None:
            if self._on_changed_callback is not None:
                self._unsubscribe(fxcorepy.O2GTableUpdateType.UPDATE)
            if self._on_deleted_callback is not None:
                self._unsubscribe(fxcorepy.O2GTableUpdateType.DELETE)
            if self._on_added_callback is not None:
                self._unsubscribe(fxcorepy.O2GTableUpdateType.INSERT)
            if self._on_status_changed_callback is not None:
                self._table.unsubscribe_status(self)

    def subscribe(self, table: fxcorepy.O2GTable = None) -> None:
        """ Subscribes a table listener to updates of a certain table.
        
            Parameters
            ----------
            table : O2GTable
                An instance of O2GTable.
        
            Returns
            -------
            None
        
        """
        if self._table is None:
            if table is None:
                raise ValueError("Table is not set")
            self._table = table
        if self._on_changed_callback is not None:
            self._table.subscribe_update(fxcorepy.O2GTableUpdateType.UPDATE, self)
        if self._on_deleted_callback is not None:
            self._table.subscribe_update(fxcorepy.O2GTableUpdateType.DELETE, self)
        if self._on_added_callback is not None:
            self._table.subscribe_update(fxcorepy.O2GTableUpdateType.INSERT, self)
        if self._on_status_changed_callback is not None:
            self._table.subscribe_status(self)
