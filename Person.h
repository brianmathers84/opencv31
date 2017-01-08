#ifndef PERSON_H
#define	PERSON_H
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <fstream>
#include <ctime>
using namespace std;
class Person
{
public:
	int _personid;
	int _isperson;
	string _firstname;
	string _surname;
	string _emailid;
	string _mobileno;
	string _soundfile;
	bool _playedsound;
	time_t _lastplayed;
	Person();
	Person(int personid, int isperson, string firstname, string surname, 
		string emailid, string mobileno, string soundfile);
	~Person();
};
#endif	/* PERSON_H */

