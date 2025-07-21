CREATE DATABASE IF NOT EXISTS rental_data;

USE rental_data;

CREATE TABLE rental_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    house_format VARCHAR(10),
    house_type VARCHAR(255),
    house_size INT,
    location VARCHAR(255),
    city VARCHAR(50),
    price INT,
    numBathrooms INT,
    SecurityDeposit INT,
    Status VARCHAR(100)
);

DROP TABLE rental_data;

SHOW TABLES;

DESC rental_data;

SELECT * FROM rental_data;

SELECT COUNT(id) FROM rental_data;

SELECT MIN(numBathrooms), MAX(numBathrooms) FROM rental_data;

SELECT DISTINCT location from rental_data;

SELECT DISTINCT house_type from rental_data;

SELECT DISTINCT house_size from rental_data;