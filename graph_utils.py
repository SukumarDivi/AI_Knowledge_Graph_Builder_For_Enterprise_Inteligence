from neo4j import GraphDatabase
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


@dataclass
class Job:
    job_id: str
    category: str
    workplace: str
    employment_type: str
    priority_class: str
    demand_score: float
    city: str
    country: str
    region: str
    department: str
    department_category: str
    is_active: bool
    text_description: str


def load_jobs_from_neo4j(uri, username, password):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    query = """
    MATCH (j:Job)-[:LOCATED_IN]->(l:Location),
          (j)-[:IN_DEPARTMENT]->(d:Department),
          (j)-[:BELONGS_TO]->(c:Category)
    OPTIONAL MATCH (j)-[:REQUIRES]->(s:Skill)
    RETURN
        j.id AS job_id,
        c.name AS category,
        j.workplace AS workplace,
        j.employment_type AS employment_type,
        j.priority_class AS priority_class,
        toFloat(j.demand_score) AS demand_score,
        l.city AS city,
        l.country AS country,
        l.region AS region,
        d.name AS department,
        d.category AS department_category,
        j.is_active AS is_active,
        collect(DISTINCT s.name) AS skills
    """
    jobs = []
    with driver.session() as session:
        for record in session.run(query):
            skills_list = record["skills"] or []
            text = (
                f"Job: {record['category']}\n"
                f"Location: {record['city']}, {record['country']} ({record['region']} region)\n"
                f"Work: {record['workplace']} {record['employment_type']}\n"
                f"Department: {record['department']} ({record['department_category']})\n"
                f"Priority: {record['priority_class']}\n"
                f"Demand Score: {record['demand_score']:.1f}/100\n"
                f"Required Skills: {', '.join(skills_list) if skills_list else 'Not specified'}"
            )
            jobs.append(Job(
                job_id=record['job_id'],
                category=record['category'],
                workplace=record['workplace'],
                employment_type=record['employment_type'],
                priority_class=record['priority_class'],
                demand_score=float(
                    record['demand_score']) if record['demand_score'] else 0.0,
                city=record['city'] or 'Unknown',
                country=record['country'] or 'Unknown',
                region=record['region'] or 'Unknown',
                department=record['department'] or 'Unknown',
                department_category=record['department_category'] or 'Unknown',
                is_active=bool(record['is_active']),
                text_description=text
            ))
    driver.close()
    return jobs


def load_graph_data(uri, username, password):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    nodes, edges = [], []
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (n)
                RETURN elementId(n) AS eid, labels(n)[0] AS label,
                       coalesce(n.id, n.name, elementId(n)) AS display_id
                LIMIT 1300
            """)
            for r in result:
                eid = str(r["eid"])
                nodes.append(
                    {"id": eid, "label": r["label"], "name": str(r["display_id"])})
        with driver.session() as session:
            result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN elementId(a) AS src, elementId(b) AS tgt, type(r) AS rel
                LIMIT 5243
            """)
            for r in result:
                edges.append(
                    {"src": str(r["src"]), "tgt": str(r["tgt"]), "rel": r["rel"]})
    finally:
        driver.close()
    return nodes, edges


def load_stats(uri, username, password):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    stats = {}
    try:
        with driver.session() as session:
            result = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt")
            stats["nodes"] = {r["label"]: r["cnt"] for r in result}
        with driver.session() as session:
            result = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt")
            stats["edges"] = {r["rel"]: r["cnt"] for r in result}
        with driver.session() as session:
            result = session.run(
                "MATCH (j:Job)-[:REQUIRES]->(s:Skill) RETURN s.name AS skill, count(j) AS cnt ORDER BY cnt DESC LIMIT 20")
            stats["top_skills"] = [(r["skill"], r["cnt"]) for r in result]
    finally:
        driver.close()
    return stats


def get_node_details_from_neo4j(uri, username, password, node_name, node_label):
    """Fetch full details of a specific node from Neo4j for the AI Agent"""
    driver = GraphDatabase.driver(uri, auth=(username, password))
    details = {"name": node_name, "label": node_label,
               "properties": {}, "relationships": []}
    try:
        with driver.session() as session:
            if node_label == "Job":
                result = session.run("""
                    MATCH (j:Job {id: $name})
                    OPTIONAL MATCH (j)-[:LOCATED_IN]->(l:Location)
                    OPTIONAL MATCH (j)-[:IN_DEPARTMENT]->(d:Department)
                    OPTIONAL MATCH (j)-[:BELONGS_TO]->(c:Category)
                    OPTIONAL MATCH (j)-[:REQUIRES]->(s:Skill)
                    RETURN j, l, d, c, collect(DISTINCT s.name) AS skills
                    LIMIT 1
                """, name=node_name)
                for r in result:
                    details["properties"] = dict(r["j"])
                    rels = []
                    if r["l"]:
                        rels.append(
                            f"LOCATED_IN -> {r['l'].get('city', '?')}, {r['l'].get('country', '?')}")
                    if r["d"]:
                        rels.append(
                            f"IN_DEPARTMENT -> {r['d'].get('name', '?')}")
                    if r["c"]:
                        rels.append(f"BELONGS_TO -> {r['c'].get('name', '?')}")
                    rels.append(
                        f"REQUIRES -> {', '.join(r['skills']) if r['skills'] else 'None'}")
                    details["relationships"] = rels
            elif node_label == "Skill":
                result = session.run("""
                    MATCH (s:Skill {name: $name})
                    OPTIONAL MATCH (j:Job)-[:REQUIRES]->(s)
                    RETURN s, count(j) AS job_count, collect(DISTINCT j.id)[0..5] AS sample_jobs
                """, name=node_name)
                for r in result:
                    details["properties"] = {
                        "name": node_name, "job_count": r["job_count"]}
                    details["relationships"] = [
                        f"REQUIRED_BY {r['job_count']} jobs", f"Sample: {', '.join(r['sample_jobs'])}"]
            elif node_label == "Location":
                result = session.run("""
                    MATCH (l:Location {city: $name})
                    OPTIONAL MATCH (j:Job)-[:LOCATED_IN]->(l)
                    RETURN l, count(j) AS job_count
                """, name=node_name)
                for r in result:
                    details["properties"] = dict(r["l"])
                    details["relationships"] = [
                        f"HAS {r['job_count']} jobs located here"]
            elif node_label == "Department":
                result = session.run("""
                    MATCH (d:Department {name: $name})
                    OPTIONAL MATCH (j:Job)-[:IN_DEPARTMENT]->(d)
                    RETURN d, count(j) AS job_count
                """, name=node_name)
                for r in result:
                    details["properties"] = dict(r["d"])
                    details["relationships"] = [
                        f"HAS {r['job_count']} jobs in this department"]
            elif node_label == "Category":
                result = session.run("""
                    MATCH (c:Category {name: $name})
                    OPTIONAL MATCH (j:Job)-[:BELONGS_TO]->(c)
                    RETURN c, count(j) AS job_count
                """, name=node_name)
                for r in result:
                    details["properties"] = dict(r["c"])
                    details["relationships"] = [
                        f"HAS {r['job_count']} jobs in this category"]
    except Exception as e:
        details["error"] = str(e)
    finally:
        driver.close()
    return details


def generate_subgraph_image(node_names_labels, edges_data, title="Knowledge Graph Subgraph"):
    """
    Generate a static PNG subgraph image using networkx + matplotlib.
    node_names_labels: list of dicts with keys 'name' and 'label'
    edges_data: list of dicts with keys 'src_name', 'tgt_name', 'rel'
    Returns: bytes of the PNG image, or None on failure
    """
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from io import BytesIO

        color_map = {
            "Job":        "#6366f1",
            "Location":   "#10b981",
            "Department": "#f59e0b",
            "Category":   "#ef4444",
            "Skill":      "#06b6d4",
        }

        G = nx.DiGraph()
        node_colors_list = []
        node_sizes_list = []

        seen_nodes = {}
        for item in node_names_labels:
            name = str(item["name"])
            label = str(item.get("label", "Job"))
            if name not in seen_nodes:
                seen_nodes[name] = label
                G.add_node(name, label=label)

        for e in edges_data:
            src = str(e.get("src_name", ""))
            tgt = str(e.get("tgt_name", ""))
            rel = str(e.get("rel", ""))
            if src and tgt and src in seen_nodes and tgt in seen_nodes:
                G.add_edge(src, tgt, rel=rel)

        for node in G.nodes():
            lbl = seen_nodes.get(node, "Job")
            node_colors_list.append(color_map.get(lbl, "#8b5cf6"))
            node_sizes_list.append(1800 if lbl == "Job" else 1200)

        if len(G.nodes()) == 0:
            return None

        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        try:
            pos = nx.spring_layout(G, k=2.5, seed=42, iterations=60)
        except Exception:
            pos = nx.circular_layout(G)

        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_color=node_colors_list,
                               node_size=node_sizes_list,
                               alpha=0.92)
        nx.draw_networkx_labels(G, pos, ax=ax,
                                font_color="white", font_size=7,
                                font_weight="bold")
        nx.draw_networkx_edges(G, pos, ax=ax,
                               edge_color="#94a3b8", alpha=0.6,
                               arrows=True, arrowsize=15,
                               connectionstyle="arc3,rad=0.1",
                               width=1.5)
        edge_labels = {(u, v): d["rel"]
                       for u, v, d in G.edges(data=True) if d.get("rel")}
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                         font_color="#94a3b8", font_size=6,
                                         bbox=dict(boxstyle="round,pad=0.2",
                                                   fc="#0d1117", ec="none", alpha=0.7))

        legend_handles = [
            mpatches.Patch(color=color_map[l], label=l)
            for l in color_map if any(seen_nodes.get(n) == l for n in G.nodes())
        ]
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper left",
                      facecolor="#1e293b", labelcolor="white",
                      fontsize=8, framealpha=0.8)

        ax.set_title(title, color="white", fontsize=13,
                     fontweight="bold", pad=14)
        ax.axis("off")
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=130,
                    facecolor="#0d1117", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as ex:
        print(f"generate_subgraph_image error: {ex}")
        return None


def build_node_subgraph_data(uri, username, password, node_name, node_label):
    """
    Fetch the immediate 1-hop neighbourhood of a node from Neo4j
    and return (node_names_labels, edges_data) suitable for generate_subgraph_image.
    """
    driver = GraphDatabase.driver(uri, auth=(username, password))
    node_names_labels = []
    edges_data = []
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (center)
                WHERE coalesce(center.id, center.name) = $name
                  AND $label IN labels(center)
                OPTIONAL MATCH (center)-[r]->(nb)
                OPTIONAL MATCH (nb2)-[r2]->(center)
                RETURN
                    center,
                    labels(center)[0]      AS center_label,
                    collect(DISTINCT {
                        nb_name:  coalesce(nb.id, nb.name),
                        nb_label: labels(nb)[0],
                        rel:      type(r),
                        dir:      'out'
                    }) AS out_rels,
                    collect(DISTINCT {
                        nb_name:  coalesce(nb2.id, nb2.name),
                        nb_label: labels(nb2)[0],
                        rel:      type(r2),
                        dir:      'in'
                    }) AS in_rels
                LIMIT 1
            """, name=node_name, label=node_label)

            for rec in result:
                node_names_labels.append(
                    {"name": node_name, "label": node_label})

                for item in (rec["out_rels"] or []):
                    nb = item.get("nb_name")
                    nbl = item.get("nb_label")
                    rel = item.get("rel")
                    if nb and nbl and rel:
                        node_names_labels.append(
                            {"name": str(nb), "label": str(nbl)})
                        edges_data.append({"src_name": node_name,
                                           "tgt_name": str(nb), "rel": rel})

                for item in (rec["in_rels"] or []):
                    nb = item.get("nb_name")
                    nbl = item.get("nb_label")
                    rel = item.get("rel")
                    if nb and nbl and rel:
                        node_names_labels.append(
                            {"name": str(nb), "label": str(nbl)})
                        edges_data.append({"src_name": str(nb),
                                           "tgt_name": node_name, "rel": rel})
    except Exception as ex:
        print(f"build_node_subgraph_data error: {ex}")
    finally:
        driver.close()

    # deduplicate nodes
    seen = set()
    deduped = []
    for n in node_names_labels:
        k = (n["name"], n["label"])
        if k not in seen:
            seen.add(k)
            deduped.append(n)
    return deduped, edges_data


def build_search_subgraph_data(uri, username, password, job_metadata_list):
    """
    Given a list of job metadata dicts (from RAG retrieval results),
    fetch their 1-hop connections from Neo4j and return subgraph data.
    """
    driver = GraphDatabase.driver(uri, auth=(username, password))
    node_names_labels = []
    edges_data = []
    job_ids = [m.get("job_id") for m in job_metadata_list if m.get("job_id")]
    if not job_ids:
        return [], []
    try:
        with driver.session() as session:
            result = session.run("""
                UNWIND $ids AS jid
                MATCH (j:Job {id: jid})
                OPTIONAL MATCH (j)-[:LOCATED_IN]->(l:Location)
                OPTIONAL MATCH (j)-[:IN_DEPARTMENT]->(d:Department)
                OPTIONAL MATCH (j)-[:BELONGS_TO]->(c:Category)
                OPTIONAL MATCH (j)-[:REQUIRES]->(s:Skill)
                RETURN
                    j.id AS job_id,
                    coalesce(l.city, 'Unknown')    AS city,
                    coalesce(d.name, 'Unknown')    AS dept,
                    coalesce(c.name, 'Unknown')    AS cat,
                    collect(DISTINCT s.name)[0..3] AS skills
                LIMIT 40
            """, ids=job_ids[:10])

            for rec in result:
                jid = str(rec["job_id"])
                city = str(rec["city"])
                dept = str(rec["dept"])
                cat = str(rec["cat"])

                node_names_labels.append({"name": jid,  "label": "Job"})
                node_names_labels.append({"name": city, "label": "Location"})
                node_names_labels.append({"name": dept, "label": "Department"})
                node_names_labels.append({"name": cat,  "label": "Category"})

                edges_data.append(
                    {"src_name": jid, "tgt_name": city, "rel": "LOCATED_IN"})
                edges_data.append(
                    {"src_name": jid, "tgt_name": dept, "rel": "IN_DEPARTMENT"})
                edges_data.append(
                    {"src_name": jid, "tgt_name": cat,  "rel": "BELONGS_TO"})

                for skill in (rec["skills"] or []):
                    if skill:
                        node_names_labels.append(
                            {"name": str(skill), "label": "Skill"})
                        edges_data.append({"src_name": jid,
                                           "tgt_name": str(skill), "rel": "REQUIRES"})

    except Exception as ex:
        print(f"build_search_subgraph_data error: {ex}")
    finally:
        driver.close()

    seen = set()
    deduped = []
    for n in node_names_labels:
        k = (n["name"], n["label"])
        if k not in seen:
            seen.add(k)
            deduped.append(n)
    return deduped, edges_data


print("graph_utils.py written!")
