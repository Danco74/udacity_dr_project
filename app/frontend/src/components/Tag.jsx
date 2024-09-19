import React from "react";

const Tag = ({ tag, value }) => {

    // Determine background and text colors based on value
    const bg_color = value == 0 ? "bg-white" : "bg-gray-600";
    const text_color = value == 0 ? "text-black" : "text-white";

    return (
        // Apply dynamic background and text colors, along with padding, border, and shadow
        <div className={`${bg_color} ${text_color} px-4 py-2 rounded-md shadow-md border border-black`}>
            {tag}  {/* Display the tag */}
        </div>
    );
};

export default Tag;
