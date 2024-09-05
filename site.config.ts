// site.config.ts
import { defineSiteConfig } from 'valaxy'

export default defineSiteConfig({
  
  lang: 'zh-CN',
  url: 'http://kiyotakali.top/',
  // 站点标题
  title: 'void’s blog',
  // 作者信息
  author: {
    avatar: 'https://avatars.githubusercontent.com/u/123150988?s=400&u=61225ea1289724b1fbeaea0089ebb453d7ab07f9&v=4',
    name: 'void'
  },
  // 站点描述
  description: '你好，我是void,欢迎',
  // 站点主题(hairy)
  theme: 'hairy',
  // or more...

  social:[
    {
      name: 'GitHub',
      link: 'https://github.com/xjtu-wjz',
      icon: 'i-ri-github-line',
      color: '#6e5494',
    },
    {
      name: 'RSS',
      link: '',
      icon: 'i-ri-rss-line',
      color: 'orange',
    },
    {
      name: 'Telegram Channel',
      link: '',
      icon: 'i-ri-telegram-line',
      color: '#0088CC',
    },
    {
      name: 'Telegram Group',
      link: 'https://t.me/blogszh',
      icon: 'i-ri:telegram-fill',
      color: '"#0088CC'
    },
    {
      name: 'E-Mail',
      link: '',
      icon: 'i-ri-mail-line',
      color: '#8E71C1',
    },
    {
      name: 'Travelling',
      link: '',
      icon: 'i-ri-train-line',
      color: 'var(--va-c-text)',
    },
    {
      name: '网易云音乐',
      link: 'https://music.163.com/#/user/home?id=1428613796',
      icon: 'i-ri-netease-cloud-music-line',
      color: '#C20C0C',
    },

  ],

  statistics: { enable: true },
  comment: { enable: true },
})