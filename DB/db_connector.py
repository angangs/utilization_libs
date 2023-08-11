import os
import pandas as pd
import sqlalchemy

# Read environment variables
os.environ['VM_IP'] = "127.0.0.1"
os.environ['PORT'] = "5000"
os.environ['DATABASE'] = "postgres"
os.environ['USER'] = "postgres"
os.environ['PASSWORD'] = "mysecretpassword"

VM_IP = os.environ["VM_IP"]
PORT = os.environ["PORT"]
DATABASE = os.environ["DATABASE"]
USER = os.environ["USER"]
PASSWORD = os.environ["PASSWORD"]


def db_connect():
    modules = "postgres+psycopg2"
    conn_str = (
        "{}://{}:{}@{}:{}/{}".format(modules, USER, PASSWORD, VM_IP, PORT, DATABASE)
    )
    print(conn_str)
    engine = sqlalchemy.create_engine(conn_str)
    postgre_sql_connection = engine.connect()
    return postgre_sql_connection


def write_data(table):
    postgre_sql_connection = db_connect()
    raw_data = pd.read_csv('poi.csv', sep=';', encoding='utf-8-sig')
    raw_data.iloc[:20000,:].to_sql(name=table, con=postgre_sql_connection, index=False, if_exists="replace", chunksize=10000,
                    method="multi")
    postgre_sql_connection.close()


def read_data(lat1, lat2, lon1, lon2):
    postgre_sql_connection = db_connect()
    df = pd.read_sql_query('SELECT * FROM "poi" WHERE "Latitude">{} and "Latitude"<{} and "Longitude">{} and "Longitude"<{}'.format(
        lat2, lat1, lon1, lon2), con=postgre_sql_connection)
    return df
