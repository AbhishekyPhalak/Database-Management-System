# Database Management System (MyDB)

A non-relational, JSON-based database management system (DBMS) built with Python. It features memory-efficient data processing, a custom query language, and advanced data operations designed to handle large datasets efficiently.

## Features
- **Non-relational JSON-based Model**: Stores and processes data in JSON format, ensuring flexibility and scalability.
- **Chunked Data Processing**: Data is processed in small, manageable chunks (as small as 1KB), resulting in faster execution, especially for massive datasets.
- **Custom Query Language (CQL)**: A natural language-like query language designed for easy data interactions and reducing query complexity by 30%.
- **Efficient Data Operations**: Implements k-way merge sort for ordering and the Nested Loop Join algorithm for joins, preventing memory overflow and reducing the system's memory footprint by 15%.

## Commands

### Database Operations
- **Create Database**: Create a new database with a specified name, file path, chunk size, and primary key.  
  `create database named MyDB with file_path=data.json, chunk_size=2, primary_key=id`
- **Switch Database**: Switch to a specific database.  
  `switch to db MyDB`
- **Exit Database**: Exit the current database.  
  `exit db`
- **Exit Application**: Exit the application.  
  `exit`
- **Delete Database**: Delete a specific database.  
  `delete database named MyDB`

### Data Operations
- **Search Data by Primary Key**: Search for data using the primary key.  
  `search data with primary id 123`
- **Insert Data**: Insert a new data record into the database.  
  `insert {"id": 124, "name": "Sample Name", "age": 30}`
- **Delete Data**: Delete data by the specified primary key.  
  `delete data where id=123`
- **Update Data**: Update data by primary key and set or remove attributes.  
  `update data where id=123 set name="New Name" remove age`

### Data Query Operations
- **Join Datasets**: Join two datasets on specified keys.  
  `join dataset1 on key1 with dataset2 on key2`
- **Select Data with Condition (Advanced)**: Select data with advanced conditions such as aggregation, grouping, and limiting results.  
  `select data with condition id>100 attributes=name,age group_by=department limit=10 aggregate=sum(salary)`
- **Select Data**: Simple select query with conditions, sorting, and limiting results.  
  `select name, age where age>25 order by age desc limit 5`

## Performance Optimizations

- **Chunked Processing**: Process large datasets in smaller, more manageable chunks to reduce memory usage and speed up execution.

- **Efficient Sorting & Joins**: Uses advanced algorithms like k-way merge sort and Nested Loop Join to optimize sorting and joining of datasets without compromising memory efficiency.