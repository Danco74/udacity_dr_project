import React from 'react'
import TagsHeatMap from '../components/TagsHeatMap'
import TagsBarChart from '../components/TagsBarChart'
import DbSummary from '../components/DbSummary'

const DbInfo = () => {
    return (
        <>
            {/* Section for displaying DB summary */}
            <div className="p-6 w-2/3 mx-auto mt-10 bg-white shadow-lg">
                <h1 className="text-2xl font-bold mb-4">DB Info</h1>
                <DbSummary />
            </div>

            {/* Section for displaying the correlation matrix (heatmap) */}
            <div className="p-6 w-2/3 mx-auto mt-10 bg-white shadow-lg">
                <h1 className="text-2xl font-bold mb-4">Correlation Matrix</h1>
                <TagsHeatMap />
            </div>

            {/* Section for displaying tag distribution (bar chart) */}
            <div className="p-6 w-2/3 mx-auto mt-10 bg-white shadow-lg">
                <h1 className="text-2xl font-bold mb-4">Tag Distribution</h1>
                <TagsBarChart />
            </div>
        </>
    )
}

export default DbInfo
