use std::{
    io::Read,
    process::{Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    layout::Flex,
    prelude::*,
    widgets::{Bar, BarChart, BarGroup, Block},
};
use rustfft::{FftPlanner, num_complex::Complex};

const RATE: usize = 44100;
const N: usize = 2048;
const BANDS: usize = 64;

fn default_monitor() -> String {
    let out = Command::new("pactl")
        .arg("get-default-sink")
        .output()
        .expect("pactl not found");
    format!("{}.monitor", String::from_utf8_lossy(&out.stdout).trim())
}

fn audio_thread(bands: Arc<Mutex<Vec<f32>>>) {
    let mut child = Command::new("parec")
        .args([
            "-d",
            &default_monitor(),
            "--format=s16le",
            "--rate=44100",
            "--channels=1",
            "--latency-msec=20",
        ])
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to start parec");
    let mut stdout = child.stdout.take().unwrap();

    let fft = FftPlanner::<f32>::new().plan_fft_forward(N);
    let window: Vec<f32> = (0..N)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / N as f32).cos())
        .collect();

    // log-spaced band edges, 40 Hz .. 16 kHz
    let edges: Vec<usize> = (0..=BANDS)
        .map(|b| {
            let f = 40.0 * (16000.0f32 / 40.0).powf(b as f32 / BANDS as f32);
            ((f * N as f32 / RATE as f32) as usize).clamp(1, N / 2 - 1)
        })
        .collect();

    let mut raw = vec![0u8; N * 2];
    loop {
        if stdout.read_exact(&mut raw).is_err() {
            break;
        }
        let mut buf: Vec<Complex<f32>> = raw
            .chunks_exact(2)
            .zip(&window)
            .map(|(b, w)| {
                let s = i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0;
                Complex::new(s * w, 0.0)
            })
            .collect();
        fft.process(&mut buf);

        let mut out = bands.lock().unwrap();
        for b in 0..BANDS {
            let (lo, hi) = (edges[b], edges[b + 1].max(edges[b] + 1));
            let peak =
                buf[lo..hi].iter().map(|c| c.norm()).fold(0.0f32, f32::max) / (N as f32 / 4.0);
            let db = 20.0 * (peak + 1e-9).log10();
            let level = ((db + 70.0) / 60.0).clamp(0.0, 1.0);
            // fast attack, slow decay
            out[b] = level.max(out[b] * 0.85);
        }
    }
}

fn main() -> std::io::Result<()> {
    let bands = Arc::new(Mutex::new(vec![0.0f32; BANDS]));
    {
        let b = bands.clone();
        thread::spawn(move || audio_thread(b));
    }

    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let mut term = Terminal::new(CrosstermBackend::new(stdout))?;

    loop {
        let data = bands.lock().unwrap().clone();
        term.draw(|f| {
            let area = f.area();

            // choose how many bars to show and how wide, based on terminal size
            let inner_w = area.width.saturating_sub(2) as usize; // minus borders
            let bar_width: usize = 2;
            let gap: usize = 1;
            let shown = (inner_w / (bar_width + gap)).clamp(1, BANDS);

            // resample the 64 bands down to `shown` bars
            let bars: Vec<Bar> = (0..shown)
                .map(|i| {
                    let idx = i * BANDS / shown;
                    let v = data[idx];
                    Bar::default()
                        .value((v * 100.0) as u64)
                        .text_value(String::new())
                        .style(Style::default().fg(Color::Rgb(204, 101, 0)))
                })
                .collect();

            // exact width the chart needs, then center it horizontally
            let chart_w = (shown * (bar_width + gap)) as u16 + 2;
            let [centered] = Layout::horizontal([Constraint::Length(chart_w)])
                .flex(Flex::Center)
                .areas(area);

            // optional: limit height and center vertically too
            let [centered] = Layout::vertical([Constraint::Length(area.height.min(40))])
                .flex(Flex::Center)
                .areas(centered);

            let chart = BarChart::default()
                .block(Block::bordered().title_alignment(Alignment::Center))
                .data(BarGroup::default().bars(&bars))
                .bar_width(bar_width as u16)
                .bar_gap(gap as u16)
                .max(100);
            f.render_widget(chart, centered);
        })?;

        if event::poll(Duration::from_millis(33))? {
            if let Event::Key(k) = event::read()? {
                if k.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(term.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}
