use std::{borrow::Cow, fmt::Debug, marker::PhantomData, sync::Arc};

use async_trait::async_trait;
use bytemuck::Pod;
use pollster::FutureExt;
use wgpu::{util::DeviceExt, Adapter, Buffer, ComputePipeline, Device, Queue, ShaderModule};

const SHADER: &str = include_str!("u32_array.wgsl");

#[derive(Debug, Clone)]
pub struct GpuDevice {
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl GpuDevice {
    pub async fn new() -> GpuDevice {
        let instance = wgpu::Instance::new(wgpu::Backends::all());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        }
    }

    pub async fn from_adapter(adapter: Adapter) -> GpuDevice {
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        }
    }

    #[inline]
    pub fn create_shader_module(&self, shader: &str) -> ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
            })
    }

    #[inline]
    pub fn create_compute_pipeline(&self, shader: &str, entry_point: &str) -> ComputePipeline {
        let cs_module = self.create_shader_module(shader);
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &cs_module,
                entry_point,
            })
    }

    #[inline]
    pub fn create_gpu_buffer_with_data(&self, data: &[impl Pod]) -> Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Values Buffer"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
    }

    #[inline]
    pub fn create_empty_buffer(&self, size: u64) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    #[inline]
    pub fn create_retrive_buffer(&self, size: u64) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    #[inline]
    pub fn create_scalar_buffer(&self, value: &impl Pod) -> Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scalar Buffer"),
                contents: bytemuck::cast_slice(&[*value]),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    pub fn clone_buffer(&self, buffer: &Buffer) -> Buffer {
        let staging_buffer = self.create_empty_buffer(buffer.size());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer.size());

        let submission_index = self.queue.submit(Some(encoder.finish()));

        self.device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));

        staging_buffer
    }
}

pub fn print_u32_array(gpu_device: &GpuDevice, data: &Buffer, name: &str) {
    let size = data.size() as wgpu::BufferAddress;

    let staging_buffer = gpu_device.create_retrive_buffer(size);
    let mut encoder = gpu_device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.copy_buffer_to_buffer(&data, 0, &staging_buffer, 0, size);

    let submission_index = gpu_device.queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    gpu_device
        .device
        .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));

    if let Some(Ok(())) = receiver.receive().block_on() {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
        println!("{} {:?}", name, result);
    } else {
        panic!("failed to run compute on gpu!");
    }
}

pub struct PrimitiveArrayGpu<T: Pod + Debug> {
    pub(crate) data: Arc<Buffer>,
    pub(crate) gpu_device: GpuDevice,
    pub(crate) phantom: PhantomData<T>,
    pub(crate) len: usize,
}

type UInt32ArrayGPU = PrimitiveArrayGpu<u32>;

impl<T> From<&[T]> for PrimitiveArrayGpu<T>
where
    T: Pod + Debug,
{
    fn from(value: &[T]) -> Self {
        let gpu_device = GpuDevice::new().block_on();

        let data = gpu_device.create_gpu_buffer_with_data(value);

        Self {
            data: Arc::new(data),
            gpu_device,
            phantom: Default::default(),
            len: value.len(),
        }
    }
}

impl<T: Pod + Debug> PrimitiveArrayGpu<T> {
    pub fn raw_values(&self) -> Option<Vec<T>> {
        let size = self.data.size() as wgpu::BufferAddress;

        let staging_buffer = self.gpu_device.create_retrive_buffer(size);
        let mut encoder = self
            .gpu_device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(&self.data, 0, &staging_buffer, 0, size);

        let submission_index = self.gpu_device.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.gpu_device
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));

        if let Some(Ok(())) = receiver.receive().block_on() {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            staging_buffer.unmap(); // Unmaps buffer from memory
            Some(result[0..self.len].to_vec())
        } else {
            panic!("failed to run compute on gpu!")
        }
    }
}

#[async_trait]
pub trait ArrowAdd<Rhs> {
    type Output;

    async fn add(&self, value: &Rhs);
}

#[async_trait]
impl ArrowAdd<UInt32ArrayGPU> for UInt32ArrayGPU {
    type Output = Self;

    async fn add(&self, value: &UInt32ArrayGPU) {
        print_u32_array(&self.gpu_device, &self.data, "via print function");
        print_u32_array(&self.gpu_device, &value.data, "via print function");
        println!("via impl function {:?}", self.raw_values().unwrap());
        println!("via impl function {:?}", value.raw_values().unwrap());
    }
}

#[tokio::main]
async fn main() {
    println!("Hello, world!");

    let values_1 = vec![1u32, 2, 3, 0, 4];
    let values_2 = vec![10u32, 20, 30, 00, 40];

    let gpu_array_1 = PrimitiveArrayGpu::<u32>::from(&values_1[..]);
    let gpu_array_2 = PrimitiveArrayGpu::<u32>::from(&values_2[..]);

    gpu_array_1.add(&gpu_array_2).await;
}
