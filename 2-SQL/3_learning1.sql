create table account(
	user_id SERIAL primary key,
	username varchar(50) unique not NULL,
	password varchar(250) NOT NULL,
	email varchar(250) UNIQUE NOT NULL,
	created_on TIMESTAMP NOT NULL,
	last_login TIMESTAMP );
	
create table job(
	job_id serial primary key,
	job_name varchar(200) unique not null);


create table account_job(
	USER_id integer references account(user_id),
	job_id integer references job(job_id),
	hire_date timestamp);

-- insert the value in the table
insert into account(username ,password,email,created_on)
values
('Ram','root','ram1@sanjivani.org.in',current_timestamp);

--how to find the data is inserted in the table or not
select * from account;

--another value is inserted 
insert into account(username ,password,email,created_on)
values
('Ram2','root','ram2@sanjivani.org.in',current_timestamp);

select * from account;

--insert into the another table
insert into job(job_name)
values
('data scientist');
--check the record are inserted or not
select * from job;

-- insert into account_job
insert into account_job(job_id, user_id,hire_date )
values
(1,1,current_timestamp);
--data are inserted in the table checking
select * from account_job;


update account 
set last_login=CURRENT_TIMESTAMP;

--returing  the affected row
select * from account;

update account 
set last_login = current_timestamp;

select * from account;

update account
set last_login=created_on
returning user_id,last_login;

select * from job;
select * from account_job;

update account_job
set hire_date=account.created_on
from account
where account_job.user_id= account.user_id;

select * from account;
select * from account_job;

update account
set last_login=current_timestamp
returning email,created_on,last_login;

Insert into job(job_name)
values ('Cowboy');

DELETE FROM job 
WHERE job_name = 'Cowboy'
RETURNING job_id,job_name;

CREATE TABLE information(
	info_id SERIAL PRIMARY KEY,
	title VARCHAR(500) NOT NULL,
	person VARCHAR(50) NOT NULL UNIQUE
);

ALTER TABLE information 
RENAME TO new_info;

select * from information;

select * from new_info;

ALTER TABLE new_info 
RENAME COLUMN person TO people;

INSERT INTO new_info 
VALUES ('some_new_title');

ALTER TABLE new_info 
ALTER COLUMN people DROP NOT NULL;

INSERT INTO new_info(title)
VALUES ('some new title');

SELECT * FROM new_info;

ALTER TABLE new_info 
DROP COLUMN people;

SELECT * FROM new_info;

ALTER TABLE new_info 
DROP COLUMN people;

ALTER TABLE new_info 
DROP COLUMN IF EXISTS people;
