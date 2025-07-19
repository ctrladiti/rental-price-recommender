CREATE DATABASE IF NOT EXISTS rental_data;

USE rental_data;

CREATE TABLE IF NOT EXISTS listings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    city VARCHAR(100),
    area VARCHAR(100),
    bhk INT,
    sqft INT,
    bathrooms INT,
    furnished_status VARCHAR(20),
    amenities TEXT,
    rent INT,
    date_listed DATE
);

CREATE TABLE rental_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    house_type VARCHAR(255),
    house_size VARCHAR(255),
    location VARCHAR(255),
    city VARCHAR(100),
    latitude FLOAT,
    longitude FLOAT,
    price DECIMAL(15,2),
    currency VARCHAR(10),
    numBathrooms INT,
    numBalconies INT,
    isNegotiable BOOLEAN,
    priceSqFt DECIMAL(10,2),
    verificationDate DATE,
    description TEXT,
    SecurityDeposit VARCHAR(100),
    Status VARCHAR(100)
);

SHOW TABLES;

DROP TABLE listings;