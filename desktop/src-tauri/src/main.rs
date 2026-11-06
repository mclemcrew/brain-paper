#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use std::{io::{BufRead, BufReader}, process::{Child, Command, Stdio}, sync::Mutex, time::Duration};
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder, webview::{DownloadEvent, NewWindowResponse}};

// External setup links open in the system browser, never inside the studio.
fn allowed_setup_link(url: &tauri::Url) -> bool {
    matches!(url.as_str(),
        "https://huggingface.co/stabilityai/stable-audio-3-optimized" |
        "https://huggingface.co/laion/clap-htsat-unfused" |
        "https://huggingface.co/settings/tokens" |
        "https://ai.google.dev/gemma/terms")
}
fn open_setup_link(url: &tauri::Url) {
    if !allowed_setup_link(url) { return; }
    #[cfg(target_os = "macos")]
    let result = Command::new("/usr/bin/open").arg(url.as_str()).spawn();
    #[cfg(target_os = "windows")]
    let result = Command::new("rundll32.exe").args(["url.dll,FileProtocolHandler", url.as_str()]).spawn();
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let result = Command::new("xdg-open").arg(url.as_str()).spawn();
    match result {
        Ok(mut child) => { std::thread::spawn(move || { let _ = child.wait(); }); }
        Err(err) => eprintln!("Could not open setup page in browser: {err}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn external_links_are_limited_to_setup_pages() {
        for url in ["https://huggingface.co/stabilityai/stable-audio-3-optimized", "https://huggingface.co/laion/clap-htsat-unfused", "https://huggingface.co/settings/tokens", "https://ai.google.dev/gemma/terms"] {
            assert!(allowed_setup_link(&url.parse().unwrap()));
        }
        for url in ["http://huggingface.co/settings/tokens", "https://huggingface.co/settings/tokens?token=secret", "https://huggingface.co.evil.example/settings/tokens", "file:///etc/passwd", "https://example.com"] {
            assert!(!allowed_setup_link(&url.parse().unwrap()));
        }
    }
}

struct Engine(Mutex<Option<Child>>);
impl Engine {
    fn stop(&self) {
        if let Some(mut child) = self.0.lock().unwrap().take() {
            drop(child.stdin.take()); // EOF asks Python to save and stop its worker.
            std::thread::spawn(move || {
                for _ in 0..50 {
                    if child.try_wait().ok().flatten().is_some() { return; }
                    std::thread::sleep(Duration::from_millis(500));
                }
                let _ = child.kill();
                let _ = child.wait();
            });
        }
    }
}
fn main() {
    let app = tauri::Builder::default().setup(|app| {
        let root = app.path().app_data_dir()?;
        std::fs::create_dir_all(&root)?;
        let binary = app.path().resource_dir()?.join("resources/ravi-local/ravi-local");
        let mut command = if cfg!(debug_assertions) && !binary.exists() {
            let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
            let mut c = Command::new(repo.join(".venv/bin/python"));
            c.current_dir(repo).args(["-m", "ravi.local.server"]);
            c
        } else { Command::new(binary) };
        let log = std::fs::File::create(root.join("startup.log"))?;
        let mut child = command.arg("--root").arg(&root).arg("--parent-stdin")
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(log).spawn()?;
        let stdout = child.stdout.take().unwrap();
        app.manage(Engine(Mutex::new(Some(child))));
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            let result = reader.read_line(&mut line).map(|_| line);
            let _ = tx.send(result);
            // Drain stdout for the lifetime of the service.
            for _ in reader.lines() {}
        });
        let line = rx.recv_timeout(Duration::from_secs(45))??;
        let value: serde_json::Value = serde_json::from_str(&line)?;
        let url: tauri::Url = value["url"].as_str().ok_or("Missing local service URL")?.parse()?;
        if url.host_str() != Some("127.0.0.1") { return Err("Invalid local service URL".into()); }
        let origin = url.origin();
        WebviewWindowBuilder::new(app, "studio", WebviewUrl::External(url))
            .title("rAvI · Composer studio").inner_size(1200.0, 860.0).min_inner_size(760.0, 600.0)
            .on_navigation(move |target| {
                if target.origin() == origin { return true; }
                open_setup_link(target);
                false
            })
            .on_new_window(|target, _| {
                open_setup_link(&target);
                NewWindowResponse::Deny
            })
            .on_download(|_, event| {
                if let DownloadEvent::Requested { destination, .. } = event {
                    let name = destination.file_name().and_then(|n|n.to_str()).unwrap_or("ravi-export.wav").to_owned();
                    match rfd::FileDialog::new().set_file_name(name).save_file() {
                        Some(path) => { *destination = path; true },
                        None => false,
                    }
                } else { true }
            }).build()?;
        Ok(())
    }).build(tauri::generate_context!()).expect("Unable to open rAvI Studio; see startup.log in the application data directory");
    app.run(|app, event| {
        if matches!(event, tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit) {
            if let Some(engine) = app.try_state::<Engine>() { engine.stop(); }
        }
    });
}
