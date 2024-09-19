import React, { useState, useEffect } from "react";

const DbSummary = () => {
    const [data, setData] = useState(null);  // State to store the fetched data
    const [loading, setLoading] = useState(true);  // State to track loading status
    const [error, setError] = useState(null);  // State to track any error during fetching

    // Fetch database summary data on component mount
    useEffect(() => {
        fetch('http://localhost:3001/db_summary')
            .then((response) => response.json())
            .then((data) => {
                console.log('DB Info:', data);
                setData(data);  // Store the fetched data in state
                setLoading(false);  // Loading complete
            })
            .catch((error) => {
                console.error('Error fetching data:', error);
                setError(error);  // Store error if fetch fails
                setLoading(false);  // Ensure loading is complete even on error
            });
    }, []);  // Empty dependency array ensures this runs only once on mount

    // Render loading state
    if (loading) {
        return <p>Loading data...</p>;
    }

    // Render error state if there's an error
    if (error) {
        return <p>Error fetching data: {error.message}</p>;
    }

    // Render the database summary data
    return (
        <>
            <ul className="list-disc pl-5">
                <li><strong>Version:</strong> {data.version}</li>
                <li><strong>Dialect:</strong> {data.dialect}</li>
                <li><strong>Driver:</strong> {data.driver}</li>
                <li><strong>Table:</strong> {data.tables[0]}</li>
            </ul>
        </>
    );
};

export default DbSummary;
