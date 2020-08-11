import React from "react"
import { NavLink } from "react-router-dom"

import headerStyles from "./header.module.scss"

const Header = () => {

  return (
    <header className={headerStyles.header}>
      <h1>
        <NavLink className={headerStyles.title} to="/show_tell">
          Connoisseur
        </NavLink>
      </h1>
      <nav>
        <ul className={headerStyles.navList}>
          <li>
            <NavLink
              className={headerStyles.navItem}
              activeClassName={headerStyles.activeNavItem}
              to="/"
              exact={true}
            >
              Home
            </NavLink>
          </li>
          <li>
            <NavLink
              className={headerStyles.navItem}
              activeClassName={headerStyles.activeNavItem}
              to="/show_and_tell_bts"
            >
              How it works
            </NavLink>
          </li>
        </ul>
      </nav>
    </header>
  )
}

export default Header
