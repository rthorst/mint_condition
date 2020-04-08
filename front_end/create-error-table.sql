/* SQL to create log files, if they do not exist */
create table if not exists error (
	error_id serial primary key,
	time_utc numeric not null,
	stack_trace varchar,
	note varchar
);

