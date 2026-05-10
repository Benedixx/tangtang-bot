from __future__ import annotations

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Cari informasi terkini di web: berita, harga, fakta, artikel, dokumentasi. "
                "Buat query yang spesifik dan presisi, bukan raw pesan user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query yang spesifik dan presisi",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_url",
            "description": (
                "Baca isi lengkap dari URL/link yang ada di pesan. "
                "Gunakan saat user share link dan ingin bot membaca isinya."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL lengkap yang akan di-scrape",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_gif",
            "description": (
                "Kirim GIF animasi sebagai ekspresi visual Tangtang. "
                "KAPAN menggunakan: saat kamu (Tangtang) punya reaksi emosi yang kuat atau situasi butuh ekspresi visual — "
                "kaget, ketawa, excited, kesel, bete, kagum, malu, salut, dll. "
                "Tidak perlu diminta user — panggil sendiri kalau momen pas. "
                "Respons boleh hanya URL GIF saja tanpa teks, atau teks singkat + URL GIF. "
                "Query HARUS dimulai dengan 'robin hsr' atau 'soundoriented', "
                "diikuti ekspresi: shocked, laughing, happy, sad, angry, excited, blushing, dancing, clapping, dll."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Query GIF, diawali 'robin hsr' atau 'soundoriented' "
                            "+ ekspresi (shocked, laughing, happy, sad, angry, excited, blushing, dancing, dll)"
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_upsert",
            "description": (
                "Simpan atau update fakta tentang user dalam memori jangka panjang. "
                "Gunakan saat user kasih info baru tentang dirinya: nama, preferensi, kebiasaan, profil."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "Fakta yang perlu disimpan, misal: 'Alice suka kopi hitam'",
                    }
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": (
                "Hapus fakta dari memori yang sudah tidak relevan, salah, atau diganti info baru. "
                "Gunakan sebelum memory_upsert jika ada fakta lama yang perlu diganti."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Kata kunci dari fakta yang ingin dihapus",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
]
