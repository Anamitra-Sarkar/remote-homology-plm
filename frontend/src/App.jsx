import React, { useEffect, useState } from "react";
import { Link, Route, Routes, useLocation } from "react-router-dom";
import { signIn, signOutUser, watchAuth } from "./firebase.js";
import HomePage from "./pages/HomePage.jsx";
import SearchPage from "./pages/SearchPage.jsx";
import MethodologyPage from "./pages/MethodologyPage.jsx";

export default function App() {
  const [user, setUser] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [authError, setAuthError] = useState(null);
  const location = useLocation();

  useEffect(
    () =>
      watchAuth((u) => {
        setUser(u);
        setAuthChecked(true);
      }),
    [],
  );

  return (
    <div className="shell">
      <nav className="navbar">
        <Link to="/" className="brand">
          <span className="brand-mark">RH</span>
          <span className="brand-name">Remote Homology</span>
        </Link>
        <div className="nav-links">
          <Link to="/" className={location.pathname === "/" ? "nav-link active" : "nav-link"}>
            Overview
          </Link>
          <Link to="/search" className={location.pathname === "/search" ? "nav-link active" : "nav-link"}>
            Search
          </Link>
          <Link to="/methodology" className={location.pathname === "/methodology" ? "nav-link active" : "nav-link"}>
            Methodology
          </Link>
        </div>
        <div className="nav-actions">
          {authChecked && !user && (
            <button className="button button-primary" onClick={() => signIn().catch((err) => setAuthError(String(err)))}>
              Sign in
            </button>
          )}
          {user && (
            <div className="nav-user">
              <span>{user.email}</span>
              <button className="button button-quiet" onClick={() => signOutUser()}>
                Sign out
              </button>
            </div>
          )}
        </div>
      </nav>
      {authError && <p className="error" style={{ padding: "8px 24px 0" }}>{authError}</p>}

      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/search" element={<SearchPage user={user} authChecked={authChecked} />} />
        <Route path="/methodology" element={<MethodologyPage />} />
      </Routes>

      <footer className="footer">
        <p>
          Research tool only. A high similarity score is evidence of a possible remote evolutionary
          relationship, not a confirmed structural or functional match — always verify against a real
          structural database before relying on any result here.
        </p>
      </footer>
    </div>
  );
}
