#pragma once
#include "stdafx.h"
#include <stdlib.h>
#include <iostream>
#include <dirent.h>
#include <string.h>
#include "mysql_connection.h"
#include "Person.h"
#include "PeopleImages.h"
#include <vector>
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
using namespace std;
class sqlCommands
{
private:
	vector<PeopleImages> _arrPeople;
	sql::Driver *_driver;
	sql::Connection *_con;
	sql::Statement *_stmt;
	sql::ResultSet *_res;

public:	
	sqlCommands();
	~sqlCommands();
	void initialiseDB();
	void readImages();
	void setPeople(vector<PeopleImages>);
	vector<PeopleImages> getPeople();
};

