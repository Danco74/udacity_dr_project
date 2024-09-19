import Tag from "./Tag";

const ResultsDisplay = (props) => {
    const { classificationResults } = props;  // Destructure classification results from props

    console.log(classificationResults);  // Log results for debugging

    return (
        <>
            {/* Container for displaying the tags */}
            <div className="p-6 w-2/3 mx-auto mt-10 bg-white rounded-lg shadow-lg">
                <h1 className="text-2xl font-bold mb-4">Tag List</h1>
                
                {/* Display tags dynamically from classification results */}
                <div className="flex flex-wrap gap-2">
                    {Object.entries(classificationResults).map(([key, value], index) => (
                        <Tag key={index} tag={key} value={value} />  // Pass key-value pairs to Tag component
                    ))}
                </div>
            </div>
        </>
    );
};

export default ResultsDisplay;


