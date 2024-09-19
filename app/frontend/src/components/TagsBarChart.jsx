import React from 'react';
import { useEffect, useState } from 'react';
import { ResponsiveBar } from '@nivo/bar';

const TagBarChart = () => {

  const [tagData, setTagData] = useState([]);  // State to store fetched tag data
  const [error, setError] = useState(null);    // State to track any errors

  // Convert the fetched object into an array for the chart
  const convertData = (dataObj) => {
    return Object.keys(dataObj).map(key => ({
      name: key,
      value: dataObj[key]
    }));
  };

  // Fetch tag distribution data on component mount
  useEffect(() => {
    fetch('http://localhost:3001/dist')
      .then((response) => response.json())
      .then((data) => {
        const convertedData = convertData(data);  // Convert the response data
        console.log('Tag Distribution:', convertedData);
        setTagData(convertedData.sort((a, b) => a.value - b.value));  // Sort by value
        console.log("Sorted Tag Data:", tagData);
      })
      .catch((error) => {
        console.error('Error fetching data:', error);
        setError(error);  // Store any error that occurs during the fetch
      });
  }, []);

  return (
    <div style={{ height: 800 }}>
      {/* Render the responsive bar chart */}
      <ResponsiveBar
        data={tagData}
        keys={['value']}  // Use 'value' field for the chart
        indexBy="name"    // Use 'name' field for labels
        layout="horizontal"
        margin={{ top: 50, right: 50, bottom: 50, left: 150 }}
        padding={0.3}
        colors="#6B7280"
        axisBottom={{
          tickSize: 5,
          tickRotation: 30
        }}
        axisLeft={{
          tickSize: 5,
          tickPadding: 5,
          tickRotation: 0,
        }}
        labelSkipWidth={12}
        labelSkipHeight={12}
        labelTextColor={{ from: 'color', modifiers: [['darker', 1.6]] }}
        animate={true}
      />
    </div>
  );
};

export default TagBarChart;
