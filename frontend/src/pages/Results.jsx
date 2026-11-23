// Results presentation view — handles single grievance card or batch results
import { ResultCard, BatchResults } from "../components.jsx";

export default function Results({ single, batch }) {
  if (single) return <ResultCard r={single} />;
  if (batch) return <BatchResults data={batch} />;
  return null;
}
