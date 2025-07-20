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
    price INT,
    numBathrooms INT,
    verificationDate VARCHAR(50),
    SecurityDeposit VARCHAR(100),
    Status VARCHAR(100)
);

SHOW TABLES;

DESC rental_data;

SELECT * FROM rental_data;

SELECT COUNT(id) FROM rental_data;

DROP TABLE rental_data;