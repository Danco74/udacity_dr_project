import React from 'react'
import Navbar from './Navbar'
import { Outlet } from 'react-router-dom'

const Layout = () => {
  return (
    <>
      {/* Render the Navbar at the top */}
      <Navbar />

      {/* Render the current route's component */}
      <Outlet />
    </>
  )
}

export default Layout
