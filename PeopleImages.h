#ifndef PEOPLEIMAGES_H
#define	PEOPLEIMAGES_H
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <fstream>
#include "Person.h"

using namespace std;
class PeopleImages
{
private:
	Person _person;
public:
	int _idpeopleimage; int _idpeoples; string _image_filename;
	PeopleImages();
	PeopleImages(Person p, int idpeopleimage, int idpeoples, string image_filename);
	Person getPerson();
	void PeopleImages::setPlayedSound(bool tf);
	~PeopleImages();
};
#endif	/* PEOPLEIMAGES_H */

