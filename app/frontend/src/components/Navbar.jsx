import React from 'react'
import { Link } from 'react-router-dom'

const Navbar = () => {
    return (
        <nav className="flex items-center justify-between bg-gray-800 p-6">
            <div className="flex items-center flex-shrink-0 text-white mr-6 space-x-8">
                {/* Project title */}
                <span className="font-semibold text-xl tracking-tight">Disaster Recovery Project</span>

                {/* Navigation Links */}
                <div className="flex space-x-4">
                    <Link to="/" className="text-white font-semibold px-3 py-2 rounded hover:text-white hover:border-white">
                        Home
                    </Link>
                    <Link to="/dbinfo" className="text-white font-semibold px-3 py-2 rounded hover:text-white hover:border-white">
                        Database Info
                    </Link>
                </div>
            </div>
        </nav>
    )
}

export default Navbar
