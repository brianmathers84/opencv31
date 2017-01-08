#include "stdafx.h"
#include "sqlCommands.h"
#include <iostream>
#include <dirent.h>
#include <string.h>
using namespace std;
sqlCommands::sqlCommands()
{
}

sqlCommands::~sqlCommands()
{
}
vector<PeopleImages> sqlCommands::getPeople()
{
	return _arrPeople;
}
void sqlCommands::setPeople(vector<PeopleImages> p)
{
	_arrPeople = p;
}
void sqlCommands::initialiseDB() {
	string server = "localhost";
	string uid = "ahome";
	string password = "P0ps!cleP0l0";
	/* Create a connection */
	_driver = get_driver_instance();
	_con = _driver->connect(server, uid, password);
}
void sqlCommands::readImages() {
	try {
		initialiseDB();
		/* Connect to the MySQL test database */
		_con->setSchema("ahome");
		
		_stmt = _con->createStatement();
		_res = _stmt->executeQuery("SELECT pi.idpeople_image, p.idpeoples, p.isperson, p.first_name, p.surname, p.email, p.mobile, pi.image_filename, p.soundfile from peoples p inner join people_image pi on p.idpeoples = pi.idpeoples ORDER BY p.idpeoples asc, pi.image_filename desc;");
		while (_res->next()) {
			Person pp(_res->getInt("idpeoples"), _res->getInt("isperson"), _res->getString("first_name"),
				_res->getString("surname"), _res->getString("email"), _res->getString("mobile"), _res->getString("soundfile"));
			PeopleImages ppi(pp, _res->getInt("idpeople_image"), _res->getInt("idpeoples"), _res->getString("image_filename"));
			_arrPeople.push_back(ppi);
		}
		_con->close();
		delete _res;
		delete _stmt;
		delete _con;
	}
	catch (sql::SQLException) {
	}
}
