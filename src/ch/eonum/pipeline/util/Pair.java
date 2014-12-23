package ch.eonum.pipeline.util;

/**
 * Generic pair/tuple class.
 * 
 * @author tim
 *
 * @param <L>
 * @param <R>
 */
public class Pair<L, R> {

	public final L left;
	public final R right;

	public Pair(L left, R right) {
		this.left = left;
		this.right = right;
	}

	@Override
	public int hashCode() {
		return left.hashCode() ^ right.hashCode();
	}

	@Override
	public boolean equals(Object o) {
		if (o == null)
			return false;
		if (!(o instanceof Pair))
			return false;
		Pair<?, ?> pairo = (Pair<?, ?>) o;
		return this.left.equals(pairo.left)
				&& this.right.equals(pairo.right);
	}

}