import React from "react";
import { Link } from "react-router-dom";

export default function HomePage() {
  return (
    <div className="container">
      <section className="hero">
        <span className="eyebrow">Evolution-aware protein search</span>
        <h1>
          Find distant protein relatives that <em>look nothing alike</em> — but share the same origin.
        </h1>
        <p className="lede">
          Two proteins can share a common evolutionary ancestor even when their sequences look almost
          completely different. Standard sequence-matching tools miss this "remote homology" — this tool
          uses a protein language model instead, and reports honestly how much better (or not) that
          actually does against a conventional sequence-similarity search.
        </p>
        <div className="hero-actions">
          <Link to="/search" className="button button-primary">
            Search a sequence &rarr;
          </Link>
          <Link to="/methodology" className="button button-quiet">
            How it's evaluated
          </Link>
        </div>
      </section>

      <section className="feature-grid">
        <FeatureCard
          index="01"
          title="Real evolutionary benchmark"
          text="Evaluated on real classified protein domains, testing exactly the hard case: relatives from the same family group that share no obvious sequence similarity."
        />
        <FeatureCard
          index="02"
          title="Compared honestly, not assumed"
          text="Every result is checked against a conventional sequence-similarity search on the same real test cases — reported as-is, whichever way it comes out."
        />
        <FeatureCard
          index="03"
          title="Confidence, not certainty"
          text="Every match comes with a similarity score and its evolutionary grouping — read as a lead worth investigating, never a confirmed answer."
        />
      </section>
    </div>
  );
}

function FeatureCard({ index, title, text }) {
  return (
    <div className="feature-card">
      <span className="feature-index">{index}</span>
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}
