SELECT * FROM customer;

SELECT customer_id,
CASE 
	WHEN (customer_id <= 100) THEN 'Premium'
	WHEN (customer_id BETWEEN 100 and 200) THEN 'Plus'
	ELSE 'Normal'
END
FROM customer;

SELECT customer_id,
CASE customer_id
	WHEN 2 THEN 'winner'
	WHEN 5 THEN 'Second Place'
	ELSE 'Normal'
END AS raffle_results
FROM customer;


SELECT 
CASE rental_rate
	WHEN 0.99 THEN 1
	ELSE 0
END 
FROM film;


SELECT 
SUM(CASE rental_rate
    WHEN 0.99 THEN 1
	ELSE 0
END) AS number_of_bargains
FROM film;

SELECT 
SUM(CASE rental_rate
    WHEN 0.99 THEN 1
	ELSE 0
END) AS bargains,
SUM(CASE rental_rate
   	WHEN 2.99 THEN 1
   ELSE 0
END) AS regular
FROM film;

SELECT 
SUM(CASE rental_rate
    WHEN 0.99 THEN 1
	ELSE 0
END) AS bargains,
SUM(CASE rental_rate
   	WHEN 2.99 THEN 1
   ELSE 0
END) AS regular,
SUM(CASE rental_rate
   WHEN 4.99 THEN 1
   ELSE 0
END) AS pqr;

