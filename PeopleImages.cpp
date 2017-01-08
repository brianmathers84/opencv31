#include "stdafx.h"
#include "PeopleImages.h"
PeopleImages::PeopleImages() {};
PeopleImages::PeopleImages(Person p, int idpeopleimage, int idpeoples, string image_filename)
{
	_person = p;
	_idpeopleimage = idpeopleimage;
	_idpeoples = idpeoples;
	_image_filename = image_filename;
}

void PeopleImages::setPlayedSound(bool tf)
{
	_person._playedsound = tf;
	if (tf)
		_person._lastplayed = time(0);
	else 
		_person._lastplayed = time(0) - (10*60);
}

Person PeopleImages::getPerson()
{
	return _person;
}
PeopleImages::~PeopleImages()
{
}
