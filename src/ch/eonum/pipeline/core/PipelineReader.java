package ch.eonum.pipeline.core;

import java.io.File;

/**
 * read a dataset in the pipeline format.
 * @author tim
 *
 */
public class PipelineReader extends DataSetReader<SparseInstance> {

	public PipelineReader(String fileName) {
		super(fileName);
	}

	public PipelineReader(File f) {
		super(f);
	}

	@Override
	protected void convertLine(DataSet<SparseInstance> data, String line) {
		data.addInstance(SparseInstance.parse(line));
	}

}
