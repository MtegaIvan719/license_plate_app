DROP TABLE IF EXISTS User;
DROP TABLE IF EXISTS LicensePlates;
-- Table for storing user information (authentication)
CREATE TABLE User (
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- Unique ID for each user
    username TEXT UNIQUE NOT NULL,       -- Username (must be unique)
    password TEXT NOT NULL               -- Hashed password
);

-- Table for logging license plate data
CREATE TABLE LicensePlates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Unique ID for each record
    start_time TEXT NOT NULL,              -- Entry/start timestamp
    end_time TEXT,                         -- Exit/end timestamp (nullable)
    license_plate TEXT NOT NULL,           -- Detected license plate text
    payment INTEGER                        -- Payment amount for the wash
);
