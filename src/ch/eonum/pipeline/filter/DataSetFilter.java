package ch.eonum.pipeline.filter;

import java.util.Collections;
import java.util.Comparator;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.SparseInstance;

/**
 * Filter configuration which can be applied to a data set.
 * 
 * @author tim
 * 
 */
public class DataSetFilter {
	public enum FilterOp {
		GT, LT, GE, LE, EQUAL, EXISTS
	}

	private String feature;
	private FilterOp filterOp;
	private double threshold;

	public DataSetFilter(String feature) {
		init(feature, FilterOp.EXISTS, 0);
	}

	public DataSetFilter(String feature, FilterOp filterOp, double threshold) {
		init(feature, filterOp, threshold);
	}

	public void init(String feature, FilterOp filterOp, double threshold) {
		this.feature = feature;
		this.filterOp = filterOp;
		this.threshold = threshold;
	}

	public DataSet<SparseInstance> filter(DataSet<SparseInstance> data) {
		DataSet<SparseInstance> filteredSet = new DataSet<SparseInstance>();
		for (SparseInstance inst : data)
			switch (this.filterOp) {
			case GT:
				if (inst.get(this.feature) > this.threshold)
					filteredSet.addInstance(inst);
				break;
			case LT:
				if (inst.get(this.feature) < this.threshold)
					filteredSet.addInstance(inst);
				break;
			case GE:
				if (inst.get(this.feature) >= this.threshold)
					filteredSet.addInstance(inst);
				break;
			case LE:
				if (inst.get(this.feature) <= this.threshold)
					filteredSet.addInstance(inst);
				break;
			case EQUAL:
				if (Math.abs(inst.get(this.feature) - this.threshold) < 0.000001)
					filteredSet.addInstance(inst);
				break;
			case EXISTS:
				if (inst.get(this.feature) > 0.000001
						|| inst.get(this.feature) < -0.000001)
					filteredSet.addInstance(inst);
				break;
			}
		return filteredSet;
	}

	/**
	 * get the upper X percents from a certain feature.
	 * 
	 * @param data
	 * @param feature
	 * @param percentage
	 *            range 0.0 - 1.0
	 * @return
	 */
	public static DataSet<SparseInstance> filterUpperPercentage(
			DataSet<SparseInstance> data, final String feature, double percentage) {
		DataSet<SparseInstance> filteredSet = new DataSet<SparseInstance>();

		Collections.sort(data, new Comparator<SparseInstance>() {
			@Override
			public int compare(SparseInstance arg0, SparseInstance arg1) {
				return arg0.get(feature) - arg1.get(feature) < 0.0 ? 1 : -1;
			}
		});

		int i = 0;
		for (SparseInstance each : data) {
			i++;
			if (data.size() * percentage < i)
				break;
			filteredSet.addInstance(each);
		}

		return filteredSet;
	}

}
