var mocker = require('mocker-data-generator').default
var util = require('util')
const express = require('express');

const app = express();

var DATA = {
    
    'zipcode': {chance: 'zip'},
    'city': {faker: 'address.city'},
    'streetname': {faker: 'address.streetName'},
    'secondaryaddress': {faker: 'address.secondaryAddress'},
    'county': {faker: 'address.county'},
    'country': {faker: 'address.country'},
    'countrycode': {faker: 'address.countryCode'},
    'state': {faker: 'address.state'},
    'stateabbr': {faker: 'address.stateAbbr'},
    'latitude': {faker: 'address.latitude'},
    'longitude': {faker: 'address.longitude'},
    'address': {faker: 'address.streetAddress'},
    'email': {faker: 'internet.email'},
    'username':{faker: 'internet.userName'},
    'password':{faker: 'internet.password'},
    'sentence':{faker: 'lorem.sentence'},
    'word' : {faker: 'lorem.word'},
    'paragraph' : {faker : 'lorem.paragraph'},
    'firstname': {faker: 'name.firstName'},
    'lastname': {faker: 'name.lastName'},
    'name': {chance: 'name'},
    'age': {chance: 'age'},
    'phonenumber': {faker: 'phone.phoneNumberFormat'},
    'date': {faker: 'date.past'}
}

app.get('/',(req,res)=>{
    let val = req.query.value;
    var d = 0;
    if (val){
        d = DATA[val];
        if(d === undefined){
            console.log("undefined detected")
            d = DATA["word"]
	}
        console.log(d)
        mocker()
        .schema("'d'", d, 1)
        .build(function(error, data) {
            if (error) {
                res.send('error');
            }
            console.log(JSON.stringify(data))
            res.send(util.inspect(JSON.stringify(data), { depth: 10 }));
        })
    }
    else{
        res.send("please specify parameter 'value'")
    }
});


app.listen(3000);
