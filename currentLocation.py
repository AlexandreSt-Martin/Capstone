from time import sleep

from geopy.geocoders import Nominatim

geoLoc = Nominatim(user_agent="GetLoc")


def getLocation():
    latitude = 45.419510
    longitude = -75.678770
    locname = None
    for i in range(10):
        result = str(latitude) + ", " + str(longitude)
        locname = geoLoc.reverse(result)
        print(locname.address)
        latitude += 1
        longitude += 1
    sleep(2)
    return locname


def displayL():
    latitude = 45.419510
    longitude = -75.678770
    displayResult = str(latitude) + ", " + str(longitude)
    displayLocname = geoLoc.reverse(displayResult)
    return displayLocname
