--day 5/9/23 8.00
--test


--
--day5/9/23 3.00
--creation of the table and its various parameter are used in it 
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