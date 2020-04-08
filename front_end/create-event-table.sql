/* SQL to create log files, if they do not exist */
create table if not exists event (
	event_id serial primary key,
	time_utc numeric not null,
	upload_format varchar,
	ebay_url varchar,
	grade_assigned varchar /* e.g. "Execellent (PSA 5)" */ 
);

