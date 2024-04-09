import React from 'react'
import {Link} from 'react-router-dom'
import styles from './Navbar.module.css'
const Navbar = () => {
  const brandStyle = {
    color: '#121212',
    textDecoration: 'none',
    fontWeight: 'bold',
    fontSize: '22px',
    display: 'flex',
    alignItems: 'center',
};
const logoText = {
  marginLeft: '10px',
};
  return (
    <>
    <nav className={`${styles.navbar} container`}>
      <Link style={brandStyle} to="/">
        <img className={styles.image} src="/images/logo.png" alt="logo"/>
        <span style={logoText}>Easy Bill</span>
      </Link>
    </nav>
    </>
  )
}

export default Navbar