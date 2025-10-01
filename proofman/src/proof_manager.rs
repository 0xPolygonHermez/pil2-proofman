use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Condvar, Mutex,
};

pub struct Counter {
    counter: AtomicUsize,
    wait_lock: Mutex<()>,
    cvar: Condvar,
    threshold: usize,
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

impl Counter {
    pub fn new() -> Self {
        Self { counter: AtomicUsize::new(0), wait_lock: Mutex::new(()), cvar: Condvar::new(), threshold: 0 }
    }

    pub fn new_with_threshold(threshold: usize) -> Self {
        Self { counter: AtomicUsize::new(0), wait_lock: Mutex::new(()), cvar: Condvar::new(), threshold }
    }

    #[inline(always)]
    pub fn increment(&self) -> usize {
        let new_val = self.counter.fetch_add(1, Ordering::Relaxed) + 1;

        if new_val >= self.threshold {
            let _guard = self.wait_lock.lock().unwrap();
            self.cvar.notify_all();
        }

        new_val
    }

    #[inline(always)]
    pub fn decrement(&self) -> usize {
        let new_val = self.counter.fetch_sub(1, Ordering::Release) - 1;

        if new_val == 0 {
            let _guard = self.wait_lock.lock().unwrap();
            self.cvar.notify_all();
        }

        new_val
    }

    pub fn wait_until_threshold_and_check_streams<F: FnMut()>(
        &self,
        mut check_streams: F,
        error: Arc<Mutex<Option<Box<dyn std::error::Error + Send + Sync>>>>,
    ) {
        let mut guard = self.wait_lock.lock().unwrap();
        loop {
            if error.lock().unwrap().is_some() {
                break;
            }
            if self.counter.load(Ordering::Acquire) >= self.threshold {
                break;
            }
            check_streams();
            let (g, _) = self.cvar.wait_timeout(guard, std::time::Duration::from_micros(100)).unwrap();
            guard = g;
        }
    }

    pub fn wait_until_threshold(&self, error: Arc<Mutex<Option<Box<dyn std::error::Error + Send + Sync>>>>) {
        let mut guard = self.wait_lock.lock().unwrap();
        while self.counter.load(Ordering::Acquire) < self.threshold && error.lock().unwrap().is_none() {
            guard = self.cvar.wait(guard).unwrap();
        }
    }

    pub fn wait_until_value_and_check_streams<F: FnMut()>(
        &self,
        value: usize,
        mut check_streams: F,
        error: Arc<Mutex<Option<Box<dyn std::error::Error + Send + Sync>>>>,
    ) {
        let mut guard = self.wait_lock.lock().unwrap();
        loop {
            if error.lock().unwrap().is_some() {
                break;
            }
            if self.counter.load(Ordering::Acquire) >= value {
                break;
            }
            check_streams();
            let (g, _) = self.cvar.wait_timeout(guard, std::time::Duration::from_micros(100)).unwrap();
            guard = g;
        }
    }

    pub fn reset(&self) {
        self.counter.store(0, Ordering::Release);
    }

    pub fn wait_until_zero(&self, error: Arc<Mutex<Option<Box<dyn std::error::Error + Send + Sync>>>>) {
        let mut guard = self.wait_lock.lock().unwrap();
        while self.counter.load(Ordering::Acquire) > 0 && error.lock().unwrap().is_none() {
            guard = self.cvar.wait(guard).unwrap();
        }
    }

    pub fn wait_until_zero_and_check_streams<F: FnMut()>(
        &self,
        mut check_streams: F,
        error: Arc<Mutex<Option<Box<dyn std::error::Error + Send + Sync>>>>,
    ) {
        let mut guard = self.wait_lock.lock().unwrap();
        loop {
            if error.lock().unwrap().is_some() {
                break;
            }
            if self.counter.load(Ordering::Acquire) == 0 {
                break;
            }
            check_streams();
            let (g, _) = self.cvar.wait_timeout(guard, std::time::Duration::from_micros(100)).unwrap();
            guard = g;
        }
    }

    #[inline(always)]
    pub fn get_count(&self) -> usize {
        self.counter.load(Ordering::Acquire)
    }
}
