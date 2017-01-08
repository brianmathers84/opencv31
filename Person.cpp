#include "stdafx.h"
#include "Person.h"
Person::Person() {}
Person::Person(int personid, int isperson, string firstname, string surname, 
	string emailid, string mobileno, string soundfile)
{
	_personid= personid;
	_isperson= isperson;
	_firstname= firstname;
	_surname= surname;
	_emailid= emailid;
	_mobileno= mobileno;
	_soundfile = soundfile;
	_playedsound = false;
	_lastplayed = time(0) - (10*60); // 10 minutes ago
}

Person::~Person()
{
}
