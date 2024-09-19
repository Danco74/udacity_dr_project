import React, { useEffect, useState, useMemo } from 'react';

const TagsHeatMap = () => {
    const [matrixData, setMatrixData] = useState({});  // State to store matrix data
    const [maxValue, setMaxValue] = useState(null);    // State to store the max value in the matrix
    const [error, setError] = useState(null);          // State to track any errors

    // Fetch the co-occurrence matrix from the server
    useEffect(() => {
        fetch('http://localhost:3001/comatrix')
            .then((response) => {
                if (!response.ok) throw new Error('Failed to fetch');
                return response.json();
            })
            .then((data) => {
                if (typeof data === 'object' && !Array.isArray(data)) {
                    setMatrixData(data);  // Set matrix data
                } else {
                    console.error('Data is not an object:', data);
                }
                // Find the maximum value in the matrix
                const allValues = Object.values(data).flatMap((row) => Object.values(row));
                const max = Math.max(...allValues);
                setMaxValue(max);
            })
            .catch((error) => {
                console.error('Error fetching data:', error);
                setError(error);
            });
    }, []);

    // Get the keys (columns) for the matrix
    const keys = useMemo(() => {
        const firstTag = Object.keys(matrixData)[0];
        return firstTag ? Object.keys(matrixData[firstTag]) : [];
    }, [matrixData]);

    // Format the matrix data for rendering
    const formattedData = useMemo(() => {
        return Object.entries(matrixData).map(([tag, values]) => ({
            id: tag,
            ...values,
        }));
    }, [matrixData]);

    // Determine cell color based on value proportion
    const getCellColor = (value) => {
        const proportion = value / maxValue;
        if (proportion > 0.02) return 'bg-gray-500';
        if (proportion > 0.01) return 'bg-gray-300';
        if (proportion > 0.005) return 'bg-gray-100';
        return 'bg-gray-200';
    };

    if (error) {
        return <div>Error loading data: {error.message}</div>;  // Display error message if fetch fails
    }

    if (!formattedData.length || !keys.length) {
        return <div>Loading...</div>;  // Display loading message if data is not ready
    }

    return (
        <div className="w-full overflow-x-auto">
            <table className="table-fixed w-full border-collapse border border-gray-300 mx-auto mt-4">
                <thead>
                    <tr>
                        <th className="border border-gray-300 px-2 py-1 text-xs w-32"></th>
                        {/* Render table headers */}
                        {keys.map((key, index) => (
                            <th key={index} className="border border-gray-300 px-1 py-1 text-xs whitespace-normal overflow-hidden text-ellipsis">{key}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {/* Render table rows */}
                    {formattedData.map((row) => (
                        <tr key={row.id}>
                            <td className="border border-gray-300 px-1 py-1 font-bold text-xs">{row.id}</td>
                            {/* Render table cells with dynamic background color */}
                            {keys.map((key, index) => (
                                <td
                                    key={index}
                                    className={`border border-gray-300 px-1 py-1 text-center text-xs ${getCellColor(row[key])}`}
                                >
                                    {row[key] || 0}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default TagsHeatMap;
