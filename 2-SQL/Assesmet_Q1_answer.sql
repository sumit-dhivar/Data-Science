--Q1
create table jobs(
	job_id serial ,
	job_title varchar(50),
	min_salary int,
	max_salary int,
	check (max_salary < 25000)
);
--Q2
alter table jobs_new
add column email varchar(30)
default = 'not available';

--Q3
alter table jobs rename to jobs_new;

--Q4 
create table location2(
	dept_name varchar(50),
	dept_location varchar(50)
);

alter table location add column dept_id serial;

drop table location2;

alter table jobs_new 
drop column dept_id;

truncate table jobs_new;

select * from jobs_new;
--Q5 
insert into jobs_new(job_title,max_salary,min_salary,dept_id) 
values('Manager',24000,5000,1);

--Q6 
alter table jobs_new 
add column dept_id int;

select jobs_new.job_id,location.dept_id from jobs_new
inner join location on jobs_new.dept_id = location.dept_id;

--Q7
select sum(max_salary) from jobs_new;
--Q8
insert into jobs_new(job_title,max_salary,min_salary,dept_id) 
values('Developer',20000,3000,2);

insert into jobs_new(job_title,max_salary,min_salary,dept_id) 
values('Team Lead',22000,4000,3);

select * from jobs_new;

select * from location;
select avg((max_salary + min_salary)/2) from jobs_new;

--Q9
insert into location(dept_name,dept_location) values('Computer','1st Floor');

insert into location(dept_name,dept_location) values('Mechanical','2nd Floor');

create table manager_details(
	manager_id serial,
	manager_name varchar(50),
	manager_details varchar(60)
);

alter table manager_details 
add column dept_id int ;

insert into manager_details(manager_name,manager_details,dept_id) 
values('Sumit','Kopargaon',2);

select manager_name from manager_details
inner join jobs_new
on manager_details.dept_id = jobs_new.dept_id;

--Q10 
select job_id from jobs_new where dept_id = (select dept_id from location where dept_name = 'Computer');