import os
import sys
import pymongo
import certifi
from motor.motor_asyncio import AsyncIOMotorClient
from src.exception import MyException
import logging
from src.constants import DATABASE_NAME_KEY, MONGODB_URL_KEY

class MongoDBClient:
    client = None

    def __init__(self, database_name: str = DATABASE_NAME_KEY):
        self.database_name = os.getenv(database_name)
        self.client = None
        self.database = None

    async def connect(self) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment variable '{MONGODB_URL_KEY}' is not set.")
                MongoDBClient.client = AsyncIOMotorClient(mongo_db_url, tlsCAFile=certifi.where())
                await MongoDBClient.client.admin.command('ping')

            self.client = MongoDBClient.client
            self.database = self.client[self.database_name]
            logging.info("MongoDB connection successful")
        except Exception as e:
            raise MyException(e, sys)