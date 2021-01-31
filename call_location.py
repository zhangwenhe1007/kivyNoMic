
def call_location(number):
    import phonenumbers
    from phonenumbers import geocoder

    if '+' in number:
        num = number
    else:
        num = '+' + number

    x = phonenumbers.parse(num, None)

    location = geocoder.description_for_number(x, 'en')
    print(num,location)

#In case the user inputs a number with +, we remove the + so that our Quebec function works (the list must start directly with region code)

    if '+' in number:
        number = number.replace('+', '')

#Location not found: return a string otherwise because Location variable itself will return None and nothing will be displayed
#Toll-free scam numbers: numbers without locations, just scam. Typically: 833, 844, 855. 888 too though not sure

    if not location:
        if list(str(number))[1:4] == ['8', '3', '3'] or list(str(number))[1:4] == ['8', '4', '4'] or list(str(number))[1:4] == ['8', '5', '5'] or list(str(number))[1:4] == ['8', '8', '8']:
            return number + " is a toll-free scam number"
        else:
            return "Cannot find the location"

    if location == 'Quebec':
        number1 = list(str(number))
        # this means the number comes from canada or the u.s.
        if number1[1:4] == ['5', '1', '4'] or number1[1:4] == ['4', '3', '8']:
            # this means the number is from montreal
            return number + " is from Montréal"
        if number1[1:4] == ["3", "5", "4"] or number1[1:4] == ['4', '5', '0'] or number1[1:4] == ['5', '7', '9']:
            return number + " is from Banlieues de Montréal"
        if number1[1:4] == ['4', '1', '8'] or number1[1:4] == ['5', '8', '1'] or number1[1:4] == ['3', '6', '7']:
            return number + " is from Québec Est"
        if number1[1:4] == ['8', '1', '9'] or number1[1:4] == ['8', '7', '3']:
            # this means that the number is from quebec
            return number + " is from Québec Nord ou Ouest"
    else:
        return number + " is from " + location
