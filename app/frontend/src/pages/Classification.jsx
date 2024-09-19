import React, { useState, useEffect } from "react";
import InputBar from '../components/InputBar';
import ResultsDisplay from '../components/ResultsDisplay';

export const Classification = () => {
    // State to hold classification results, initialized with default values
    const [classificationResults, setClassificationResults] = useState({
        "aid_centers": 0,
        "aid_related": 0,
        "buildings": 0,
        "clothing": 0,
        "cold": 0,
        "death": 0,
        "direct_report": 0,
        "earthquake": 0,
        "electricity": 0,
        "fire": 0,
        "floods": 0,
        "food": 0,
        "hospitals": 0,
        "infrastructure_related": 0,
        "medical_help": 0,
        "medical_products": 0,
        "military": 0,
        "missing_people": 0,
        "money": 0,
        "offer": 0,
        "other_aid": 0,
        "other_infrastructure": 0,
        "other_weather": 0,
        "refugees": 0,
        "related": 0,
        "request": 0,
        "search_and_rescue": 0,
        "security": 0,
        "shelter": 0,
        "shops": 0,
        "storm": 0,
        "tools": 0,
        "transport": 0,
        "water": 0,
        "weather_related": 0
    });

    // Function to handle input changes and send input to the server
    const handleInputChange = async (inputValue) => {
        try {
            const response = await fetch('http://localhost:3001/go', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: inputValue }),
            });

            // Wait for server response and update the classification results
            const data = await response.json();
            console.log('DATA: ' + data);
            setClassificationResults(data);

        } catch (error) {
            console.error('Error sending input to the server:', error);  // Handle errors
        }
    };

    return (
        <>
            {/* Render input bar */}
            <div className="mt-20">
                <InputBar onInputChange={handleInputChange} />
            </div>

            {/* Render classification results */}
            <ResultsDisplay className="flex" classificationResults={classificationResults} />
        </>
    );
};

export default Classification;
