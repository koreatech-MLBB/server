from multiprocessing import Semaphore, shared_memory, sharedctypes
import ctypes
import time
import random
import pickle
import pymysql as sql

class Error(Exception):
    def __init__(self, message="", hint=""):
        '''
        :param message:
        :param hint:
        '''
        self.message = message
        self.hint = hint
        super().__init__(self.message)

    def __str__(self):
        '''
        :return: string representation of the Error
        '''
        return "Error message: " + self.message + "\nHint: " + self.hint
class DBConnection:
    def __init__(self, user: str, password: str, database: str, shared_memory_name: str, semaphore: Semaphore):
        '''
        :param user:
        :param password:
        :param database:
        :param shared_memory_name:
        :param semaphore:
        '''
        self.user = user
        self.database = database
        self.password = password
        self.host = 'localhost'
        self.charset = 'utf8'
        self.conn = sql.connect(host=self.host, user=self.user, password=self.password, database=self.database,
                           charset=self.charset)
        self.shared_memory = shared_memory.SharedMemory(name=shared_memory_name)
        self.semaphore = semaphore

    def __str__(self):
        '''
        :return: string representation of the DBConnection
        '''
        return f"host: {self.host}\nuser: {self.user}\npassword: {self.password}\ndatabase: {self.database}\ncharset: {self.charset}"

    def __del__(self):
        '''
        :explain: delete DBConnection
        '''
        self.conn.close()

    def select(self, table: str, condition: list = [], order: list = [()]) -> tuple:
        '''
        select(table='test')\n
        select(table='test', order={'id1': 1})\n
        select(table='test', condition=["id1", "id2"])\n
        select(table='test', condition=["id1"], order=[('id1', '=', 2)]\n
        :param table:
        :param condition: default = []
        :param order:  default = [()]
        :return: tuple
        :raises Error: pymysql exceptions
        '''
        cursor = self.conn.cursor()
        res = None
        if len(condition) == 0 and len(order) == 0:
            sql = f"select * from {table}"
            try:
                cursor.execute(sql)
            except Exception as e:
                raise Error(e.__str__(), "select 오류")
            res = cursor.fetchall()
        elif len(condition) == 0 and len(order) != 0:
            sql = f"select * from {table} where"
            for condition, value in order.items():
                if type(value) == type("v"):
                    sql += f" {condition} = '{value}' AND"
                else:
                    sql += f" {condition} = {value} AND"
            sql = sql.strip(" AND")
            try:
                cursor.execute(sql)
            except Exception as e:
                raise Error(e.__str__(), "select 오류")
            res = cursor.fetchall()
        elif len(condition) != 0 and len(order) != 0:
            sql = "select "
            for c in condition:
                sql += f"{c}, "
            sql = sql.rstrip(", ") + f" from {table} where"
            for c, o, v in order:
                if type(v) == type("v"):
                    sql += f" {c}{o}'{v}' AND"
                else:
                    sql += f" {c}{o}{v} AND"
            sql = sql.rstrip(" AND")
            try:
                cursor.execute(sql)
            except Exception as e:
                raise Error(e.__str__(), "select 오류")
            res = cursor.fetchall()
        elif len(condition) != 0 and len(order) == 0:
            sql = "select "
            for c in condition:
                sql += f"{c}, "
            sql = sql.rstrip(", ") + f" from {table}"
            try:
                cursor.execute(sql)
            except Exception as e:
                raise Error(e.__str__(), "select 오류")
            res = cursor.fetchall()
        self.conn.commit()
        return res

    def insert(self, table: str, condition: list = [], values: list = [[]]) -> bool:
        '''
        insert(table="test", condition=["id1", "id2"], values=[[1, 2]])\n
        insert(table="test", condition=["id1", "id2"], values=[[1, 1], [2, 1], [3, 2], [4, 2]])
        :param table:
        :param condition: default = []
        :param values: default = [[]]
        :return: bool
        :raises Error: if condition is null or values is null, if condition and values are not same length, pymysql exceptions
        '''
        cursor = self.conn.cursor()
        if len(condition) == 0 and len(values) == 0:
            raise Error(message="condition과 values의 값이 없습니다.", hint="condition, values")
        elif len(condition) == 0 and len(values) != 0:
            raise Error(message="condition의 값이 없습니다.", hint="condition")
        elif len(condition) != 0 and len(values) == 0:
            raise Error(message="values의 값이 없습니다", hint="values")
        else:
            for i in range(len(values)):
                if len(condition) != len(values[i]):
                    raise Error(message="condition의 갯수와 values의 갯수가 일치하지 않습니다.", hint=f"values[{i}]")
            sql = f"insert into {table} ("
            for x in condition:
                sql += f"{x}, "
            sql = sql.rstrip(", ") + ") values"
            for i in range(len(values)):
                sql += "("
                for x in values[i]:
                    if type(x) == type("x"):
                        sql += f"'{x}', "
                    else:
                        sql += f"{x}, "
                sql = sql.rstrip(", ") + "), "
            sql = sql.rstrip(", ")
            try:
                cursor.execute(sql)
            except Exception as e:
                raise Error(e.__str__(), "insert 오류")
            self.conn.commit()
            return True

    def delete(self, table: str, order: dict = {}) -> bool:
        '''
        delete(table="test", order="{"id": 1, "id2": "test"})
        :param table:
        :param order: default = {}
        :return: bool
        :raises Error: if order is null, pymysql exceptions
        '''
        cursor = self.conn.cursor()
        if len(order) == 0:
            raise Error(message="order가 비어있습니다.", hint="order")
        else:
            sql = f"delete from {table} where"
            for condition, value in order.items():
                if type(value) == type("v"):
                    sql += f" {condition}='{value}' AND"
                else:
                    sql += f" {condition}={value} AND"
            sql = sql.rstrip(" AND")
        try:
            cursor.execute(sql)
        except Exception as e:
            raise Error(e.__str__(), "delete 오류")

        self.conn.commit()
        return True

    def update(self, table: str, condition: list = [], values: list = [], order: list = [()]) -> bool:
        '''
        update(table="test", condition=["id1", "id2"], values=["val1", 1], order=[("id1", "=", "id"), ...])
        :param table:
        :param condition: default = []
        :param values: default = []
        :param order: default = [()]
        :return: bool
        :raises Error: if condition or values or order is null, pymysql exceptions
        '''
        cursor = self.conn.cursor()
        err = [False, False, False]
        errFlag = False
        if len(condition) == 0:
            err[0] = "condition"
        if len(order) == 0:
            err[1] = "order"
        if len(values) == 0:
            err[2] = "values"
        msg = ""
        for x in err:
            if x != False:
                errFlag = True
                msg += x + ", "
        msg = msg.rstrip(", ")
        if errFlag:
            raise Error(message=msg + "가 비어있습니다", hint=msg)
        else:
            if len(condition) != len(values):
                raise Error(message="condition의 갯수와 values의 갯수가 일치하지 않습니다.", hint=f"values")
            sql = f"update {table} set "
            for c, v in zip(condition, values):
                if type(v) == type("v"):
                    sql += f"{c}='{v}', "
                else:
                    sql += f"{c}={v}, "
            sql = sql.rstrip(", ") + " where"
            for c, o, v in order:
                if type(v) == type("v"):
                    sql += f" {c}{o}'{v}' AND"
                else:
                    sql += f" {c}{o}{v} AND"
            sql = sql.rstrip(" AND")
            try:
                cursor.execute(sql)
            except Exception as e:
                raise Error(e.__str__(), "udpate 오류")
            self.conn.commit()
            return True

    def run(self):
        while True:
            # db_shared_dict = sharedctypes.RawArray(ctypes.c_char, self.shared_memory.buf)
            # db_shared_dict = db_shared_dict.value.decode('utf-8')
            self.semaphore.acquire()
            db_shared_bytes = self.shared_memory.buf.tobytes()
            db_shared_dict = pickle.loads(db_shared_bytes)
            if db_shared_dict["func"] == "select":
                print(db_shared_dict)
                # select 로직 작성
            elif db_shared_dict["func"] == "insert":
                print(db_shared_dict)
                # insert 로직 작성
            elif db_shared_dict["func"] == "update":
                print(db_shared_dict)
                # update 로직 작성
            elif db_shared_dict["func"] == "delete":
                print(db_shared_dict)
                # delete 로직 작성
            else:
                print("None")
                # 그 외 상황 로직 작성
            # db_shared_dict["func"] = random.choice(["select", "insert", "update", "delete"])
            # db_shared_bytes = pickle.dumps(db_shared_dict)
            # self.shared_memory.buf[:len(db_shared_bytes)] = db_shared_bytes
            # 위 작업이 끝난 후 공유 메모리를 비우는 로직 작성
            self.semaphore.release()
            time.sleep(0.5)
