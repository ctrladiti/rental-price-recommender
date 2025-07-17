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

SHOW TABLES;
