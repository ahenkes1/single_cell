//! Generic fixed-size ring buffer for short-term memory.

/// A fixed-size circular buffer that overwrites old elements when full.
///
/// Used for storing recent sensor experiences without heap allocation.
#[derive(Debug, Clone)]
pub struct RingBuffer<T, const N: usize> {
    buffer: [T; N],
    head: usize,
    len: usize,
}

impl<T: Default + Copy, const N: usize> Default for RingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Default + Copy, const N: usize> RingBuffer<T, N> {
    /// Creates a new empty ring buffer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: [T::default(); N],
            head: 0,
            len: 0,
        }
    }

    /// Pushes a new item to the buffer, overwriting the oldest if full.
    pub fn push(&mut self, item: T) {
        self.buffer[self.head] = item;
        self.head = (self.head + 1) % N;
        if self.len < N {
            self.len += 1;
        }
    }

    /// Returns the number of items currently in the buffer.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the buffer is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Gets an item by index (0 = oldest, len-1 = newest).
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        // Calculate actual position: start from oldest element
        let start = if self.len < N { 0 } else { self.head };
        let actual_index = (start + index) % N;
        Some(&self.buffer[actual_index])
    }

    /// Returns the most recently added item.
    #[must_use]
    pub fn last(&self) -> Option<&T> {
        if self.len == 0 {
            return None;
        }
        let last_index = if self.head == 0 { N - 1 } else { self.head - 1 };
        Some(&self.buffer[last_index])
    }

    /// Returns an iterator over items from oldest to newest.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        RingBufferIter {
            buffer: self,
            current: 0,
        }
    }

    /// Clears the buffer.
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }
}

/// Iterator over ring buffer elements.
struct RingBufferIter<'a, T, const N: usize> {
    buffer: &'a RingBuffer<T, N>,
    current: usize,
}

impl<'a, T: Default + Copy, const N: usize> Iterator for RingBufferIter<'a, T, N> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.buffer.len {
            return None;
        }
        let item = self.buffer.get(self.current);
        self.current += 1;
        item
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_len() {
        let mut buf: RingBuffer<i32, 4> = RingBuffer::new();
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());

        buf.push(1);
        assert_eq!(buf.len(), 1);
        buf.push(2);
        buf.push(3);
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn test_overflow() {
        let mut buf: RingBuffer<i32, 3> = RingBuffer::new();
        buf.push(1);
        buf.push(2);
        buf.push(3);
        buf.push(4); // Overwrites 1

        assert_eq!(buf.len(), 3);
        assert_eq!(*buf.get(0).unwrap(), 2); // Oldest is now 2
        assert_eq!(*buf.get(1).unwrap(), 3);
        assert_eq!(*buf.get(2).unwrap(), 4); // Newest
    }

    #[test]
    fn test_get_ordering() {
        let mut buf: RingBuffer<i32, 4> = RingBuffer::new();
        buf.push(10);
        buf.push(20);
        buf.push(30);

        assert_eq!(*buf.get(0).unwrap(), 10); // Oldest
        assert_eq!(*buf.get(2).unwrap(), 30); // Newest
        assert!(buf.get(3).is_none());
    }

    #[test]
    fn test_last() {
        let mut buf: RingBuffer<i32, 4> = RingBuffer::new();
        assert!(buf.last().is_none());

        buf.push(1);
        assert_eq!(*buf.last().unwrap(), 1);

        buf.push(2);
        buf.push(3);
        assert_eq!(*buf.last().unwrap(), 3);
    }

    #[test]
    fn test_iter() {
        let mut buf: RingBuffer<i32, 4> = RingBuffer::new();
        buf.push(1);
        buf.push(2);
        buf.push(3);

        let collected: Vec<_> = buf.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn test_iter_after_overflow() {
        let mut buf: RingBuffer<i32, 3> = RingBuffer::new();
        buf.push(1);
        buf.push(2);
        buf.push(3);
        buf.push(4);
        buf.push(5);

        let collected: Vec<_> = buf.iter().copied().collect();
        assert_eq!(collected, vec![3, 4, 5]);
    }

    #[test]
    fn test_clear() {
        let mut buf: RingBuffer<i32, 4> = RingBuffer::new();
        buf.push(1);
        buf.push(2);
        buf.clear();

        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }
}
