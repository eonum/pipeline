package ch.eonum.pipeline.analysis;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.LinkedHashMap;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.SparseInstance;

/**
 * <p>
 * Analyze the results of a classification task.
 * </p>
 * 
 * <p>
 * Recognition rate. Recognition rate per class, class distributions, confusion
 * matrix etc..
 * </p>
 * 
 * @author tim
 * 
 */
public class ClassificationAnalyzer<E extends Instance> {

	private Map<String, Double> totalMetrics;
	private Map<String, Instance> metricsPerClass;

	/**
	 * 
	 * 
	 * @param data
	 *            labeled data set after classification
	 * @param fileName
	 *            file name for the CSV file, where all analysis results will be
	 *            written to.
	 * @throws IOException
	 */
	public void analyze(DataSet<E> data, String fileName) throws IOException {
		totalMetrics = new LinkedHashMap<String, Double>();
		metricsPerClass = new LinkedHashMap<String, Instance>();

		totalMetrics.put("N", (double) data.size());
		int correct = 0;
		for (Instance i : data)
			if (i.groundTruth != null && i.groundTruth.equals(i.label))
				correct++;

		totalMetrics.put("RecognitionRate", correct / (double) data.size()
				* 100.0);

		Set<String> classes = data.collectClasses();
		Map<String, Integer> indicesByClassName = new HashMap<String, Integer>();
		Map<Integer, String> classNamesByIndex = new HashMap<Integer, String>();
		int[][] confusionMatrix = new int[classes.size() + 1][classes.size() + 1];
		
		int j = 0;
		for (String i : classes){
			metricsPerClass.put(i, new SparseInstance(i, i,
					new HashMap<String, Double>()));
			indicesByClassName.put(i, j);
			classNamesByIndex.put(j, i);
			j++;
		}

		for (Instance i : data) {
			Instance metrics = metricsPerClass.get(i.groundTruth);
			metrics.put("N", metrics.get("N") + 1.);
			for (String c : classes) {
				Instance m = metricsPerClass.get(c);
				m.put("classProb",
						m.get("classProb") + i.getResult("classProb" + c));
			}
			
			int y = indicesByClassName.get(i.label) == null ? confusionMatrix.length - 1 : 
				indicesByClassName.get(i.label);
			int x = indicesByClassName.get(i.groundTruth) == null ? confusionMatrix.length - 1
					: indicesByClassName.get(i.groundTruth);
			confusionMatrix[x][y]++;

			if (i.groundTruth != null && i.groundTruth.equals(i.label))
				metrics.put("correct", metrics.get("correct") + 1.);
			else if (metricsPerClass.containsKey(i.label))
				metricsPerClass.get(i.label).put("falsePositive",
						metricsPerClass.get(i.label).get("falsePositive") + 1.);

		}

		for (String c : classes) {
			Instance metrics = metricsPerClass.get(c);
			metrics.put("RecognitionRate",
					metrics.get("correct") / metrics.get("N") * 100.0);
			metrics.put("classDistribution", metrics.get("N") / data.size()
					* 100.0);
			metrics.put("classProb", metrics.get("classProb") / data.size()
					* 100.0);
		}

		String[] columns = new String[] { "N", "RecognitionRate",
				"classDistribution", "classProb", "correct", "falsePositive" };

		/** write results. */
		PrintStream ps = new PrintStream(new File(fileName));
		/** BOM marker for Excel! */
		ps.write('\uFEFF');
		ps.println();
		ps.println("Total;" + data.size() + ";Recognition Rate;"
				+ totalMetrics.get("RecognitionRate"));
		ps.println();
		ps.print("class;");
		for (String c : columns)
			ps.print(c + ";");
		ps.println();
		for (String cl : classes) {
			ps.print(cl + ";");
			for (String c : columns)
				ps.print(metricsPerClass.get(cl).get(c) + ";");
			ps.println();
		}
		
		ps.println();ps.println();
		ps.println("Confusion matrix (x-axis: ground truth, y-axis: prediction");
		ps.print(";");
		for(int i = 0; i < classes.size(); i++)
			ps.print(classNamesByIndex.get(i) + ";");
		ps.println();
		for(int y = 0; y < classes.size(); y++){
			ps.print(classNamesByIndex.get(y) + ";");
			for(int x = 0; x < classes.size(); x++)
				ps.print(confusionMatrix[x][y] + ";");
			ps.println();
		}

		ps.close();
	}

	public Map<String, Instance> getMetricsByClass() {
		return this.metricsPerClass;
	}

}
