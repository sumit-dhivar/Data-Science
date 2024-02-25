create table account(
user_id serial primary key,
	username varchar(50) unique not null,
	password varchar(50) not null,
	email varchar(250) unique not null,
	last_login timestamp
);

create table job(
job_id serial primary key,
	job_name varchar(200) unique not null
);