package ch.eonum.pipeline.transformation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.SparseInstance;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

/**
 * Support class for Principle Components Analysis. This does some light
 * lifting, but the real work is done in the Jama code.
 * 
 * @author Gabe Johnson <johnsogg@cmu.edu>
 */
public class PCA extends Transformer<SparseInstance> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();

	static {
		PARAMETERS.put("k", "dimensionality/number of principal components"
					+ " on which the data is to be reduced. -1 being the original dimensionality (default: -1)");
	}

	private Matrix covMatrix;
	private EigenvalueDecomposition eigenstuff;
	private double[] eigenvalues;
	private Matrix eigenvectors;
	private SortedSet<PrincipleComponent> principleComponents;
	private double[] means;
	private Features featuresObject;

	public PCA(DataSet<SparseInstance> data, Features features) {
		this.featuresObject = features;
		this.setSupportedParameters(PCA.PARAMETERS);
		this.putParameter("k", -1.0);
		this.prepare(data);
	}

	@Override
	public void prepare(DataSet<SparseInstance> data) {
		double[][] input = data.asDoubleArrayMatrix(featuresObject);
		means = new double[input[0].length];
		double[][] cov = getCovariance(input, means);
		covMatrix = new Matrix(cov);
		eigenstuff = covMatrix.eig();
		eigenvalues = eigenstuff.getRealEigenvalues();
		eigenvectors = eigenstuff.getV();
		double[][] vecs = eigenvectors.getArray();
		int numComponents = eigenvectors.getColumnDimension(); // same as num
																// rows.
		principleComponents = new TreeSet<PrincipleComponent>();
		for (int i = 0; i < numComponents; i++) {
			double[] eigenvector = new double[numComponents];
			for (int j = 0; j < numComponents; j++) {
				eigenvector[j] = vecs[i][j];
			}
			principleComponents.add(new PrincipleComponent(eigenvalues[i],
					eigenvector));
		}
	}

	@Override
	public void extract() {
		super.extract();
		int k = (int) this.getDoubleParameter("k");
		if (k < 0)
			k = this.principleComponents.size();
		List<PrincipleComponent> mainComponents = this.getDominantComponents(k);
		Matrix features = PCA.getDominantComponentsMatrix(mainComponents);
		Matrix featuresXpose = features.transpose();
		double[][] matrixAdjusted = PCA.getMeanAdjusted(
				dataSet.asDoubleArrayMatrix(featuresObject), this.getMeans());
		Matrix adjustedInput = new Matrix(matrixAdjusted);
		Matrix xformedData = featuresXpose.times(adjustedInput.transpose());
		xformedData = xformedData.transpose();

		int i = 0;
		for (SparseInstance each : this.dataSet) {
			double[] row = xformedData.getArray()[i];
			Map<String, Double> v = new HashMap<String, Double>();
			for (int j = 0; j < row.length; j++)
				v.put(String.valueOf(j), row[j]);
			each.put(v);
			i++;
		}
	}

	public double[] getMeans() {
		return means;
	}

	/**
	 * Subtracts the mean value from each column. The means must be precomputed,
	 * which you get for free when you make a PCA instance (just call
	 * getMeans()).
	 * 
	 * @param input
	 *            Some data, where each row is a sample point, and each column
	 *            is a dimension.
	 * @param mean
	 *            The means of each dimension. This could be computed from
	 *            'input' directly, but for efficiency's sake, it should only be
	 *            done once and the result saved.
	 * @return Returns a translated matrix where each cell has been translated
	 *         by the mean value of its dimension.
	 */
	public static double[][] getMeanAdjusted(double[][] input, double[] mean) {
		int nRows = input.length;
		int nCols = input[0].length;
		double[][] ret = new double[nRows][nCols];
		for (int row = 0; row < nRows; row++) {
			for (int col = 0; col < nCols; col++) {
				ret[row][col] = input[row][col] - mean[col];
			}
		}
		return ret;
	}

	/**
	 * Returns the top n principle components in descending order of relevance.
	 */
	public List<PrincipleComponent> getDominantComponents(int n) {
		List<PrincipleComponent> ret = new ArrayList<PrincipleComponent>();
		int count = 0;
		for (PrincipleComponent pc : principleComponents) {
			ret.add(pc);
			count++;
			if (count >= n) {
				break;
			}
		}
		return ret;
	}

	public static Matrix getDominantComponentsMatrix(
			List<PrincipleComponent> dom) {
		int nRows = dom.get(0).eigenVector.length;
		int nCols = dom.size();
		Matrix matrix = new Matrix(nRows, nCols);
		for (int col = 0; col < nCols; col++) {
			for (int row = 0; row < nRows; row++) {
				matrix.set(row, col, dom.get(col).eigenVector[row]);
			}
		}
		return matrix;
	}

	public static class PrincipleComponent implements
			Comparable<PrincipleComponent> {
		public double eigenValue;
		public double[] eigenVector;

		public PrincipleComponent(double eigenValue, double[] eigenVector) {
			this.eigenValue = eigenValue;
			this.eigenVector = eigenVector;
		}

		public int compareTo(PrincipleComponent o) {
			int ret = 0;
			if (eigenValue > o.eigenValue) {
				ret = -1;
			} else if (eigenValue < o.eigenValue) {
				ret = 1;
			}
			return ret;
		}

		public String toString() {
			String eigenV = "";
			for (double e : eigenVector)
				eigenV += ", " + e;
			return "Principle Component, eigenvalue: " + eigenValue
					+ ", eigenvector: [" + eigenV + "]";
		}
	}

	public static double[][] getCovariance(double[][] input, double[] meanValues) {
		int numDataVectors = input.length;
		int n = input[0].length;

		double[] sum = new double[n];
		double[] mean = new double[n];
		for (int i = 0; i < numDataVectors; i++) {
			double[] vec = input[i];
			for (int j = 0; j < n; j++) {
				sum[j] = sum[j] + vec[j];
			}
		}
		for (int i = 0; i < sum.length; i++) {
			mean[i] = sum[i] / numDataVectors;
		}

		double[][] ret = new double[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = i; j < n; j++) {
				double v = getCovariance(input, i, j, mean);
				ret[i][j] = v;
				ret[j][i] = v;
			}
		}
		if (meanValues != null) {
			System.arraycopy(mean, 0, meanValues, 0, mean.length);
		}
		return ret;
	}

	/**
	 * Gives covariance between vectors in an n-dimensional space. The two input
	 * arrays store values with the mean already subtracted.
	 */
	private static double getCovariance(double[][] matrix, int colA, int colB,
			double[] mean) {
		double sum = 0;
		for (int i = 0; i < matrix.length; i++) {
			double v1 = matrix[i][colA] - mean[colA];
			double v2 = matrix[i][colB] - mean[colB];
			sum = sum + (v1 * v2);
		}
		int n = matrix.length;
		double ret = (sum / (n - 1));
		return ret;
	}

}
